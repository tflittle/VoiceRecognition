#include <ESP_I2S.h>

#define SAMPLE_RATE 16000

#define MIC_CLK  42
#define MIC_DATA 41

I2SClass I2S;

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println("Starting microphone test...");

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
        Serial.println("I2S initialization failed!");

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("Microphone initialized.");
}

void loop()
{
    int sample = I2S.read();

    if (sample != 0 && sample != -1 && sample != 1)
    {
        Serial.println(sample);
    }
}