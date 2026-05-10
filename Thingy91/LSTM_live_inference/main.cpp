#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>


#include "ml.h"

LOG_MODULE_REGISTER(ml_main, 5);


int main() {
    k_sleep(K_SECONDS(3));
    LOG_INF("Edge Impulse standalone inferencing (Zephyr)");
    ml_init();
}
