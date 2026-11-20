#include <ESP_I2S.h>

#define SAMPLE_RATE 16000
#define RECORD_SECONDS 1
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)

#define MIC_CLK 42
#define MIC_DATA 41
#define LIGHT_PIN 3

I2SClass I2S;

int16_t audioBuffer[NUM_SAMPLES];

void recordAudio();
void sendAudio();

void setup()
{
    Serial.begin(921600);
    delay(1000);

    // XIAO ESP32S3 Sense onboard PDM microphone
    I2S.setPinsPdmRx(MIC_CLK, MIC_DATA);

    bool success = I2S.begin(
        I2S_MODE_PDM_RX,
        SAMPLE_RATE,
        I2S_DATA_BIT_WIDTH_16BIT,
        I2S_SLOT_MODE_MONO
    );

    if (!success)
    {
        Serial.println("I2S_INIT_FAILED");

        while (true)
        {
            delay(1000);
        }
    }

    pinMode(LIGHT_PIN, OUTPUT);
    digitalWrite(LIGHT_PIN, LOW);

    Serial.println("READY");
}

void loop()
{
    if (Serial.available() > 0)
    {
        char command = Serial.read();

        if (command == 'r')
        {
            digitalWrite(LIGHT_PIN, HIGH);
            recordAudio();
            sendAudio();
            digitalWrite(LIGHT_PIN, LOW);
        }
    }
}

void recordAudio()
{
    size_t bytesRead = 0;
    size_t totalBytes = sizeof(audioBuffer);

    while (bytesRead < totalBytes)
    {
        size_t received = I2S.readBytes(
            ((char *)audioBuffer) + bytesRead,
            totalBytes - bytesRead
        );

        bytesRead += received;
    }
}

void sendAudio()
{
    Serial.write(
        (uint8_t *)audioBuffer,
        sizeof(audioBuffer)
    );

    Serial.flush();
}