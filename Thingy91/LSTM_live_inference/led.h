#ifdef __cplusplus
extern "C" {
#endif

#ifndef LED_H__
#define LED_H__

#include <stdint.h>

enum led_color {
    RED,
    GREEN,
    BLUE
};

int configure_leds();
int set_led(enum led_color led, bool state);

#endif /* LED_H__ */

#ifdef __cplusplus
}
#endif

