// hotas.c:         Thrustmaster T.Flight HOTAS One reader
// AUTHOR:          DANIEL DESAI
// UPDATED:         2026-05-10
// VERSION:         0.1.0
//
// Reads /dev/input/js0 (or argv[1]) using the Linux joydev kernel
// interface (js_event structs).  No libraries required beyond libc.
//
// Outputs one line per 20 ms (50 Hz) to stdout:
//   roll pitch thrust yaw tilt manual\n
//
// All values are normalized floats in [-1.0, 1.0] except:
//   thrust  — rescaled to [0.0, 1.0]
//   tilt    — rescaled to [0.0, 1.0]  (0=hover, 1=cruise)
//   manual  — 0.0 or 1.0  (toggled by trigger, starts at 1.0)
//
// Axis mapping (verified with jstest /dev/input/js0):
//   axis 0  stick X   left/right  → roll    (right = +1)
//   axis 1  stick Y   fwd/back    → pitch   (pull back = +1; inverted)
//   axis 2  throttle  fwd/back    → thrust  (fwd = +1; rescaled 0-1)
//   axis 3  twist Rz              → yaw     (right = +1)
//   axis 4  rocker    left/right  → tilt    (right=cruise; rescaled 0-1)
//
// Build:
//   gcc -O2 -o controls/hotas controls/hotas.c
//
// Run standalone (calibration):
//   ./controls/hotas [/dev/input/js0]
//
// Permission fix if device not readable:
//   sudo usermod -aG input $USER   (log out and back in)
//   or one-shot: sudo chmod a+r /dev/input/js0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <linux/joystick.h>   /* js_event, JSIOCGAXES, JSIOCGBUTTONS */
#include <sys/ioctl.h>

/* ── Axis indices (jstest axis numbers) ─────────────────────── */
#define AXIS_ROLL     0
#define AXIS_PITCH    1
#define AXIS_THROTTLE 2
#define AXIS_ROCKER   7   /* rest varies; +32767=cruise, -32767=hover */
#define AXIS_YAW      5
#define NUM_AXES      10   /* device has 10; we only use 0–4 */

/* Push-forward direction: -1 if fwd=negative raw, +1 if fwd=positive raw */
#define THROTTLE_DIR  (-1)

#define NUM_BUTTONS   15

/* ── Deadzone ────────────────────────────────────────────────── */
#define DEADZONE 0.04f

static float apply_deadzone(float v)
{
    float av = v < 0 ? -v : v;
    if (av < DEADZONE) return 0.0f;
    float scaled = (av - DEADZONE) / (1.0f - DEADZONE);
    return v < 0 ? -scaled : scaled;
}

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Normalise raw joydev value (−32767…+32767) to −1…+1 */
static float norm(int raw)
{
    return clampf((float)raw / 32767.0f, -1.0f, 1.0f);
}

int main(int argc, char *argv[])
{
    const char *device = argc > 1 ? argv[1] : "/dev/input/js0";

    int fd = open(device, O_RDONLY);   /* blocking — for init drain */
    if (fd < 0) {
        if (errno == EACCES)
            fprintf(stderr,
                "hotas: permission denied on %s\n"
                "  Fix: sudo usermod -aG input $USER  (then log out/in)\n"
                "  Or:  sudo chmod a+r %s\n", device, device);
        else
            fprintf(stderr, "hotas: cannot open %s: %s\n",
                    device, strerror(errno));
        return 1;
    }

    /* Print a ready line so the Julia parent knows we're connected */
    {
        char name[128] = "Unknown";
        ioctl(fd, JSIOCGNAME(sizeof(name)), name);
        int naxes = 0, nbtns = 0;
        ioctl(fd, JSIOCGAXES,    &naxes);
        ioctl(fd, JSIOCGBUTTONS, &nbtns);
        fprintf(stderr, "hotas: connected — %s (%d axes, %d buttons)\n",
                name, naxes, nbtns);
        fflush(stderr);
    }

    /* ── State ───────────────────────────────────────────────── */
    int   axes[NUM_AXES];
    int   btns[NUM_BUTTONS];
    int   cruise_mode = 0;     /* rocker: 0=hover, 1=cruise              */
    int   hover_hold  = 0;     /* btn 6: toggle altitude hold            */
    int   brakes      = 0;     /* btn 7: toggle wheel brakes             */
    int   autoland    = 0;     /* btn 1: toggle autoland/emergency land  */
    float trim        = 0.0f;  /* pitch trim bias, ±1.0 (btn 4/5)       */

    memset(axes, 0, sizeof(axes));
    memset(btns, 0, sizeof(btns));

    /* Drain JS_EVENT_INIT events — one per axis, sent immediately on open.
     * Read exactly naxes events (all init), then switch to non-blocking. */
    {
        int naxes = 0;
        ioctl(fd, JSIOCGAXES, &naxes);
        struct js_event e;
        for (int i = 0; i < naxes; i++) {
            if (read(fd, &e, sizeof(e)) == sizeof(e))
                if ((e.type & JS_EVENT_INIT) && e.number < NUM_AXES)
                    axes[e.number] = e.value;
        }
    }

    /* Switch to non-blocking for the main loop */
    {
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    /* ── Main loop: read events, emit state at 50 Hz ─────────── */
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 20000000L };  /* 20 ms */

    for (;;) {
        /* Drain all pending events (non-blocking) */
        struct js_event e;
        while (read(fd, &e, sizeof(e)) == sizeof(e)) {
            int type = e.type & ~JS_EVENT_INIT;
            if (type == JS_EVENT_AXIS && e.number < NUM_AXES)
                axes[e.number] = e.value;
            else if (type == JS_EVENT_BUTTON && e.number < NUM_BUTTONS) {
                int prev = btns[e.number];
                btns[e.number] = e.value;
                /* btn 2: autoland toggle — arms on press */
                if (e.number == 2 && e.value == 1) {
                    autoland = !autoland;
                    fprintf(stderr, "hotas: autoland %s\n", autoland ? "ARMED" : "DISARMED");
                    fflush(stderr);
                }
                /* btn 7: wheel brakes — active while held */
                if (e.number == 7) {
                    brakes = e.value;
                    fprintf(stderr, "hotas: brakes %s\n", brakes ? "ON" : "OFF");
                    fflush(stderr);
                }
                (void)prev;
            }
        }

        /* Normalisation (verified from jstest at physical rest):
         *   roll    (0): rest=0,      -1=left,      +1=right
         *   pitch   (1): rest=0,      inverted: pull-back=+1=nose up
         *   throttle(2): rest=0,      push-fwd=+1 (inverted hardware)
         *   rocker  (7): -32767=hover, +32767=cruise (threshold at 0)
         *   yaw     (5): rest=0,      twist-left=-1, twist-right=+1
         *   tilt: 0=hover, 1=cruise — rocker left half=hover, right half=cruise
         */
        float roll   = apply_deadzone(norm(axes[AXIS_ROLL]));
        float pitch  = apply_deadzone(-norm(axes[AXIS_PITCH]));
        float thrust = clampf(THROTTLE_DIR * norm(axes[AXIS_THROTTLE]), 0.0f, 1.0f);
        float yaw    = apply_deadzone(norm(axes[AXIS_YAW]));

        /* Rocker: +32767=cruise, -32767=hover.
         * Hysteresis: switch to cruise above +8000, back to hover below -8000.
         * Prevents oscillation when rocker rests near centre. */
        int new_cruise = cruise_mode ?
            (axes[AXIS_ROCKER] > -8000) :   /* stay cruise unless clearly left */
            (axes[AXIS_ROCKER] >  8000);    /* enter cruise only when clearly right */
        if (new_cruise != cruise_mode) {
            cruise_mode = new_cruise;
            fprintf(stderr, "hotas: mode → %s\n", cruise_mode ? "CRUISE" : "HOVER");
            fflush(stderr);
        }
        float tilt = (float)cruise_mode;



        /* Trim: btn 4 held = nose up (+), btn 5 held = nose down (−).
         * Ramps at ~0.1/s (0.002 per 20 ms step), clamped to ±1.0. */
        if (btns[4]) trim = clampf(trim + 0.002f, -1.0f, 1.0f);
        if (btns[5]) trim = clampf(trim - 0.002f, -1.0f, 1.0f);

        printf("%.4f %.4f %.4f %.4f %.4f %.4f %d %d\n",
               roll, pitch, thrust, yaw, tilt, trim, brakes, autoland);
        fflush(stdout);

        nanosleep(&ts, NULL);
    }

    close(fd);
    return 0;
}
