// Streams a test signal over serial for uartscope.
// One value per line at a fixed sample rate: 10 Hz sine + 50 Hz "mains hum" + noise.
//
// Run:  python -m uartscope --port <your port> --fs 500 --filter notch --cutoff 50
//
// To plot a real sensor instead, replace the signal line with e.g.
//   float v = analogRead(A0);
//
// Note: at 115200 baud the link carries about 11500 characters/s, so keep
// fs * (characters per line) well below that.

const float FS_HZ = 500.0;
const unsigned long PERIOD_US = (unsigned long)(1000000.0 / FS_HZ);

void setup() {
  Serial.begin(115200);
}

void loop() {
  static unsigned long next_us = micros();
  static unsigned long n = 0;

  if ((long)(micros() - next_us) >= 0) {
    next_us += PERIOD_US;
    float t = n++ / FS_HZ;
    float v = sin(2 * PI * 10.0 * t)
            + 0.5 * sin(2 * PI * 50.0 * t)
            + random(-100, 101) / 500.0;
    Serial.println(v, 4);
  }
}
