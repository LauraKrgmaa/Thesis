#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/gpio.h>
#include "led.h"

LOG_MODULE_REGISTER(ml, 5);

#define RED_LED_NODE    DT_ALIAS(led0)
#define GREEN_LED_NODE  DT_ALIAS(led1)
#define BLUE_LED_NODE   DT_ALIAS(led2)
#define SW0_NODE        DT_ALIAS(sw0)

#define GREEN_LED_PWM_NODE  DT_ALIAS(pwm_led1)

#define PWM_PERIOD      1024
#define LIGHTNESS_MAX   UINT16_MAX

static const struct gpio_dt_spec green_led  = GPIO_DT_SPEC_GET(GREEN_LED_NODE, gpios);
static const struct gpio_dt_spec red_led    = GPIO_DT_SPEC_GET(RED_LED_NODE, gpios);
static const struct gpio_dt_spec blue_led   = GPIO_DT_SPEC_GET(BLUE_LED_NODE, gpios);
static const struct gpio_dt_spec button     = GPIO_DT_SPEC_GET(SW0_NODE, gpios);

static const struct pwm_dt_spec green_led_pwm  = PWM_DT_SPEC_GET(GREEN_LED_PWM_NODE);

int configure_button() {
    int ret;

	ret = gpio_pin_configure_dt(&button, GPIO_INPUT);
	if (ret < 0) {
        LOG_ERR("error nupu konfimisega");
        return -1;
	}
    
    return 0;
}

int configure_leds() {
    int ret;

    ret = gpio_pin_configure_dt(&red_led, GPIO_OUTPUT);
	if (ret < 0) {
        LOG_ERR("error red_led konfimisega");
		return -1;
	}
	ret = gpio_pin_configure_dt(&green_led, GPIO_OUTPUT);
	if (ret < 0) {
        LOG_ERR("error green_led konfimisega");
		return -1;
	}
	ret = gpio_pin_configure_dt(&blue_led, GPIO_OUTPUT);
	if (ret < 0) {
        LOG_ERR("error blue_led konfimisega");
		return -1;
	}

    return 0;
}

int set_led(enum led_color led, bool state) {
    switch (led) {
        case RED:
            gpio_pin_set_dt(&red_led, state);
            break;
        case GREEN:
            gpio_pin_set_dt(&green_led, state);
            break;
        case BLUE:
            gpio_pin_set_dt(&blue_led, state);
            break;
    }
    return 0;
}

void set_pwm_led(int brightness) {
    uint32_t scaled_lvl = (PWM_PERIOD * brightness) / LIGHTNESS_MAX;
    pwm_set_dt(&green_led_pwm, PWM_USEC(PWM_PERIOD), PWM_USEC(scaled_lvl));
}

/*
LOG_INF("testing pwm");
for (int a = 0; a < 65000; a += 1000) {
    LOG_INF("pwm: %d", a);
    set_pwm_led(a);
    k_sleep(K_MSEC(300));
}
*/
