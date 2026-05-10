#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/logging/log.h>
#include <string.h>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"
#include "led.h"    
// Circular buffer - CIRC_BUF_SIZE is already total floats (e.g. 300 = 100 samples * 3 axes)
#define CIRC_BUF_SIZE EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE
#define NUM_CHANNELS 6 
#define SAMPLES_IN_WINDOW (CIRC_BUF_SIZE / NUM_CHANNELS) // actual number of XYZ samples

static float circ_buf[CIRC_BUF_SIZE];
static volatile uint32_t circ_write_idx = 0; // counts in samples (each sample = 3 floats)
static volatile bool circ_buf_ready = false;
static const double mean_accel = 4.2270617485;
static const double mean_gyro = -0.0190564748;
static const double std_accel = 5.0296134949;
static const double std_gyro = 0.9469125867;
static const double max_accel = 3.0590102672576904;
static const double max_gyro = 4.630462646484375;
LOG_MODULE_REGISTER(ml, 3); 
K_SEM_DEFINE(data_sem, 0, 1);
K_SEM_DEFINE(init_done_sem, 0, 1);
K_SEM_DEFINE(timer_sem, 0, 1);
K_MUTEX_DEFINE(buffer_mutex);
static const struct device *sensor_dev = NULL; 
static float sample[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static float single_reading[6]; //change to 3 when 3 axis
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
struct sensor_value accel[3];
struct sensor_value gyro[3];
struct k_timer mytimer;
//remove accel when using only gyro
const static enum sensor_channel sensor_channels[] = {
    SENSOR_CHAN_ACCEL_X,
	SENSOR_CHAN_ACCEL_Y,
	SENSOR_CHAN_ACCEL_Z,
    SENSOR_CHAN_GYRO_X,
    SENSOR_CHAN_GYRO_Y,
    SENSOR_CHAN_GYRO_Z
};


double normalize_gyro(double raw_val) {
    double val = (raw_val - mean_gyro) / std_gyro;
    return ((2 * val) / max_gyro);
}

double normalize_accel(double raw_val) {
    double val = (raw_val - mean_accel) / std_accel;
    return ((2 * val) / max_accel);
}
int ml_init() {
	struct sensor_value acc_freq = { .val1 = 10, .val2 = 0 };
    const struct device *i2c_dev;
    int ret;
    LOG_INF("Waiting for system stabilization...");
    k_sleep(K_SECONDS(2));

	if (sensor_dev == NULL) {
        sensor_dev = DEVICE_DT_GET_ONE(invensense_mpu6050);
    }

    if (!sensor_dev) {
        LOG_ERR("Could not get MPU6050 device pointer! Check Devicetree overlay and alias. Exiting.");
        return -ENODEV; 
    }

    while (true) {
        i2c_dev = DEVICE_DT_GET(DT_NODELABEL(i2c2));
        if (!device_is_ready(i2c_dev)) {
            LOG_ERR("I2C device (DT_NODELABEL(i2c2)) is not ready. Retrying in 2s...");
            k_sleep(K_SECONDS(2));
            continue; 
        }
        k_msleep(1000); // delay for i2c

        LOG_INF("Attempting to wake up MPU6050 from sleep...");
        bool wake_command_succeeded = false;
        for (int i = 0; i < 10; i++) { 
            k_msleep(100); 
            ret = i2c_reg_write_byte(i2c_dev, 0x68, 0x6B, 0x00); // wakeup command
            if (ret == 0) {
                LOG_INF("MPU6050 wake-up command successful on attempt %d.", i + 1);
                wake_command_succeeded = true;
                break; // success
            } else {
                LOG_WRN("MPU6050 wake-up command failed (attempt %d): %d. Retrying wake command...", i + 1, ret);
            }
        }
        k_msleep(100);

        if (!wake_command_succeeded) {
            LOG_ERR("Failed to send MPU6050 wake-up command after multiple attempts. Retrying entire config loop in 2s.");
            k_sleep(K_SECONDS(2));
            continue; 
        }
        k_msleep(200); // for stabilization

        
        if (!device_is_ready(sensor_dev)) {
            LOG_WRN("MPU6050 Zephyr driver is NOT ready yet (device_is_ready returned false). Retrying entire config loop in 1s.");
            k_sleep(K_SECONDS(1)); 
            sensor_dev = NULL; 
            sensor_dev = DEVICE_DT_GET_ONE(invensense_mpu6050);
            continue; 
        }
        LOG_INF("MPU6050 Zephyr driver reports device is READY! Connection established.");

        // Configure sensor attributes
        

        ret = sensor_attr_set(sensor_dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &acc_freq);
        if (ret < 0) {
            LOG_WRN("Could not set MPU6050 sensor reading frequency (%d), using default. This is not critical, but noted.", ret);
            uint8_t div = 19;  // For ~50 Hz: (1000 / 50) - 1
            ret = i2c_reg_write_byte(i2c_dev, 0x68, 0x19, div);  // SMPLRT_DIV reg
            if (ret == 0) {
            LOG_INF("Set SMPLRT_DIV to %d for ~50 Hz", div);
            }else{
                LOG_INF("Could not set SMPLRT_DIV to %d for ~50 Hz", div);
            }
        } else {
            LOG_INF("Set MPU6050 sensor frequency to %d Hz.", acc_freq.val1);
        
        }

        LOG_INF("MPU6050 configuration complete and sensor is ready.");
        break; 
    }

    configure_leds();
    k_sem_give(&init_done_sem); 

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        LOG_ERR("The size of your 'features' array is not correct. Expected %d items, but had %u", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    } else {
        LOG_INF("Features array size is correct (%u items).", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
        return 0;
    }


}

void classification_results(ei_impulse_result_t *result, int64_t inference_duration_ms) {
    // all classes and results
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    //    LOG_INF("%s: %.3f ", result->classification[ix].label, result->classification[ix].value);
    }
    //LOG_INF("\n");  

    size_t max_ix = 0;
    float max_value = result->classification[0].value;
    for (size_t ix = 1; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result->classification[ix].value > max_value) {
            max_value = result->classification[ix].value;
            max_ix = ix;
        }
    }

    // turn led on  based on confidence
    const char *best_label = result->classification[max_ix].label;
    if (max_value > 0.80f) {
        if (strcmp(best_label, "Normal_walk_right") == 0) {
            set_led(GREEN, true);
            set_led(RED, false);
            set_led(BLUE, false);
        } else if (strcmp(best_label, "Steppage Gait_right") == 0) {
            set_led(GREEN, false);
            set_led(RED, true);
            set_led(BLUE, false);
        } else if(strcmp(best_label, "idle") == 0){
            set_led(GREEN, false);
            set_led(RED, false);
            set_led(BLUE, true);

        }else {
            set_led(GREEN, false);
            set_led(RED, false);
            set_led(BLUE, false);
        }
    } else {
        set_led(GREEN, false);
        set_led(RED, false);
        set_led(BLUE, false);
    }
}



void sample_for_timer(){
    static uint32_t samples_since_ml = 0;
    if (sensor_sample_fetch(sensor_dev) < 0) {
            LOG_ERR("MPU6050 sensor sample update error");
            k_msleep(20);
            //break;
        }
        sensor_channel_get(sensor_dev, SENSOR_CHAN_ACCEL_XYZ, accel); //comment out when using gyro only
        sensor_channel_get(sensor_dev, SENSOR_CHAN_GYRO_XYZ, gyro);
        double norm_x_ac = normalize_accel(sensor_value_to_double(&accel[0]));
        double norm_y_ac = normalize_accel(sensor_value_to_double(&accel[1]));
        double norm_z_ac = normalize_accel(sensor_value_to_double(&accel[2]));
        double norm_x_gy = normalize_gyro(sensor_value_to_double(&gyro[0]));
        double norm_y_gy = normalize_gyro(sensor_value_to_double(&gyro[1]));
        double norm_z_gy = normalize_gyro(sensor_value_to_double(&gyro[2]));

        k_mutex_lock(&buffer_mutex, K_FOREVER); // protect during writing
        uint32_t idx = (circ_write_idx % SAMPLES_IN_WINDOW) * 6; //change to 3 if only gyro
        circ_buf[idx + 0] = norm_x_ac;
        circ_buf[idx + 1] = norm_y_ac;
        circ_buf[idx + 2] = norm_z_ac;
        circ_buf[idx + 3] = norm_x_gy;
        circ_buf[idx + 4] = norm_y_gy;
        circ_buf[idx + 5] = norm_z_gy;

        circ_write_idx++;
        k_mutex_unlock(&buffer_mutex);
        samples_since_ml++;
        if (samples_since_ml >= 10) {
        k_sem_give(&data_sem);
        samples_since_ml = 0;
    }

}

static void mytimer_l(struct k_timer *d){
    //do something
    k_sem_give(&timer_sem);
}
void get_sample(void *p1, void *p2, void *p3) {
    k_sem_take(&init_done_sem, K_FOREVER);
    //LOG_INF("Filling buffer for 2000ms...");
    //LOG_INF("CIRC_BUF_SIZE: %d, SAMPLES_IN_WINDOW: %d", CIRC_BUF_SIZE, SAMPLES_IN_WINDOW);

    int64_t start_time = k_uptime_get();
    uint32_t samples_collected = 0;
    uint32_t report_interval = 100;

    // Part 1: fill for 1 second
    while (k_uptime_get() - start_time < 1000) {
        if (sensor_sample_fetch(sensor_dev) < 0) {
            LOG_ERR("MPU6050 sensor sample update error");
            k_msleep(20);
             continue;
        }

        sensor_channel_get(sensor_dev, SENSOR_CHAN_ACCEL_XYZ, accel); //comment out when using gyro only
        sensor_channel_get(sensor_dev, SENSOR_CHAN_GYRO_XYZ, gyro);
        double norm_x_ac = normalize_accel(sensor_value_to_double(&accel[0]));
        double norm_y_ac = normalize_accel(sensor_value_to_double(&accel[1]));
        double norm_z_ac = normalize_accel(sensor_value_to_double(&accel[2]));
        double norm_x_gy = normalize_gyro(sensor_value_to_double(&gyro[0]));
        double norm_y_gy = normalize_gyro(sensor_value_to_double(&gyro[1]));
        double norm_z_gy = normalize_gyro(sensor_value_to_double(&gyro[2]));
        // idx is the base float index for this sample (each sample = 3 floats)
        uint32_t idx = (circ_write_idx % SAMPLES_IN_WINDOW) * 6;
        //circ_buf[idx + 0] = sensor_value_to_double(&gyro[0]);
        //circ_buf[idx + 1] = sensor_value_to_double(&gyro[1]);
        //circ_buf[idx + 2] = sensor_value_to_double(&gyro[2]);
        circ_buf[idx + 0] = norm_x_ac;
        circ_buf[idx + 1] = norm_y_ac;
        circ_buf[idx + 2] = norm_z_ac;
        circ_buf[idx + 3] = norm_x_gy;
        circ_buf[idx + 4] = norm_y_gy;
        circ_buf[idx + 5] = norm_z_gy;

        circ_write_idx++;
        k_msleep(20);
    }

    //LOG_INF("Buffer filled. circ_write_idx = %u. Starting ML.", circ_write_idx);
    circ_buf_ready = true;
    k_sem_give(&data_sem);
    k_timer_init(&mytimer, mytimer_l, NULL); //expiry function is mytimer_l
    k_timer_start(&mytimer, K_SECONDS(0), K_MSEC(20)); //delay and period

    // Part 2: keep writing forever
    uint32_t total_samples = 0;
    int64_t last_report_time = k_uptime_get();

    while(true){
       k_sem_take(&timer_sem, K_FOREVER);
       sample_for_timer();
       int64_t now = k_uptime_get();
       total_samples++;
       if (now - last_report_time >= 5000) {
           float actual_hz = (float)total_samples / ((float)(now - last_report_time) / 1000.0f);
           //LOG_INF("ACTUAL SAMPLE RATE: %.2f Hz", actual_hz);
           
           // Reset counters for next 5-second window
           total_samples = 0;
           last_report_time = now;
       }
    }
    
}


void ml_run(void *p1, void *p2, void *p3) {
    while (1) {
        // 1. Wait for the 200ms trigger (10 samples)
        k_sem_take(&data_sem, K_FOREVER);
        k_mutex_lock(&buffer_mutex, K_FOREVER);
        
        // head_sample is the index of the OLDEST sample in the buffer
        uint32_t head_sample = circ_write_idx % SAMPLES_IN_WINDOW;
        uint32_t head_float = head_sample * 6; //change to 3 if only gyro
        
        // How many samples exist from the head to the end of the physical array
        uint32_t samples_to_copy_first = SAMPLES_IN_WINDOW - head_sample;
        uint32_t floats_to_copy_first = samples_to_copy_first * 6;//change to 3 when gyro only

        // Copy from the head to the end of the circ_buf
        memcpy(features, 
               &circ_buf[head_float], 
               floats_to_copy_first * sizeof(float));

        // Copy from the start of circ_buf to the head (the wrap-around part)
        if (head_float > 0) {
            memcpy(&features[floats_to_copy_first], 
                   &circ_buf[0], 
                   head_float * sizeof(float));
        }
        k_mutex_unlock(&buffer_mutex);

        // prepare signal
        signal_t signal;
        numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

        // run inference
        ei_impulse_result_t result = { 0 };
        int64_t start_time = k_uptime_get();
        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
        int64_t end_time = k_uptime_get();

        LOG_INF("Inference took %d ms (DSP: %d ms, NN: %d ms)",
                (int32_t)(end_time - start_time),
                result.timing.dsp,
                result.timing.classification);

        if (res == 0) {
            classification_results(&result, end_time - start_time);
        }
        //backlog prevention

        k_sem_reset(&data_sem);
    }
}
    //start threads code
    //K thread define takes name, stack_size, entry function name,p1,
    //p2,p3,priority (lower number is higher priority),options,delay
K_THREAD_DEFINE(sensor_sample_thread, 1536, get_sample, NULL, NULL, NULL, 1, 0, 0);
K_THREAD_DEFINE(ml_inference_thread, 4096, ml_run, NULL, NULL, NULL, 10, 0, 0);
