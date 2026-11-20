#include <ESP_I2S.h>
#include <math.h>
#include "ModelWeights.h"

// ============================================================
// MICROPHONE / AUDIO SETTINGS
// ============================================================

#define SAMPLE_RATE 16000

// XIAO ESP32S3 Sense onboard PDM microphone
#define MIC_CLK  42
#define MIC_DATA 41

#define RED_LED 2
#define GREEN_LED 1
#define LIGHT_PIN 3

// Read 20 ms at a time
#define BLOCK_SIZE 320

// Model expects exactly 1 second
#define AUDIO_SAMPLES 16000

// Tune this based on microphone RMS
#define VOICE_THRESHOLD 1400.0f

#define PRETRIGGER_SAMPLES 1600

int16_t preTriggerBuffer[PRETRIGGER_SAMPLES];

int preTriggerIndex = 0;
bool preTriggerFull = false;
// ============================================================
// MFCC SETTINGS
// Must match Python training code
// ============================================================

#define FRAME_LENGTH 400
#define HOP_LENGTH 160
#define FFT_SIZE 512

#define MEL_FILTERS 26
#define MFCC_COEFFICIENTS 13

// floor((16000 - 400) / 160) + 1 = 98
#define NUM_FRAMES 98

float fftReal[FFT_SIZE];
float fftImag[FFT_SIZE];

float powerSpectrum[FFT_SIZE / 2 + 1];
float logMelBuffer[MEL_FILTERS];


// ============================================================
// GLOBAL VARIABLES
// ============================================================

I2SClass I2S;

// 20 ms microphone block
int16_t blockBuffer[BLOCK_SIZE];

// Complete 1-second recording
int16_t audioBuffer[AUDIO_SAMPLES];

// MFCC result
float mfcc[NUM_FRAMES][MFCC_COEFFICIENTS];

// Mel filter locations
int melBins[MEL_FILTERS + 2];

// Neural-network buffers
float modelInput[MODEL_INPUT_SIZE];
float hiddenLayer[MODEL_HIDDEN_SIZE];

// Recording state
bool recording = false;
int recordIndex = 0;


// ============================================================
// FUNCTION PROTOTYPES
// ============================================================

void readAudioBlock();

float calculateRMS(
    int16_t *samples,
    int count
);

void startRecording();

void addBlockToRecording();

void finishRecording();

float hzToMel(float hz);
float melToHz(float mel);

void createMelBins();

void applyHammingWindow(
    float *frame
);

void runFFT(
    float *real,
    float *imag,
    int n
);

void calculatePowerSpectrum(
    float *real,
    float *imag,
    float *power
);

void calculateMelEnergies(
    float *power,
    float *logMel
);

void calculateDCT(
    float *logMel,
    float *mfccFrame
);

void calculateMFCC(
    int16_t *audio,
    float outputMFCC[NUM_FRAMES][MFCC_COEFFICIENTS]
);

float relu(float x);
float sigmoid(float x);

void flattenMFCC(
    float inputMFCC[NUM_FRAMES][MFCC_COEFFICIENTS]
);

void normalizeModelInput();

void runHiddenLayer();

float runOutputLayer();

float runInference(
    float inputMFCC[NUM_FRAMES][MFCC_COEFFICIENTS]
);

void classifyVoice();


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(921600);
    delay(1000);

    pinMode(LIGHT_PIN, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    pinMode(RED_LED, OUTPUT);
    digitalWrite(LIGHT_PIN, LOW);
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, LOW);

    // Configure onboard PDM microphone
    I2S.setPinsPdmRx(
        MIC_CLK,
        MIC_DATA
    );

    bool success = I2S.begin(
        I2S_MODE_PDM_RX,
        SAMPLE_RATE,
        I2S_DATA_BIT_WIDTH_16BIT,
        I2S_SLOT_MODE_MONO
    );

    if (!success)
    {
        Serial.println("Microphone initialization failed!");

        while (true)
        {
            delay(1000);
        }
    }

    // Mel filters do not change, so calculate once
    createMelBins();

    Serial.println("Voice recognition ready.");
    Serial.println("Listening...");

}


// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    // Read next 20 ms microphone block
    readAudioBlock();


    // ========================================================
    // WAITING FOR VOICE
    // ========================================================

    if (!recording)
    {
        // Save this block into our rolling history
        updatePreTriggerBuffer();

        float rms = calculateRMS(
            blockBuffer,
            BLOCK_SIZE
        );

        if (rms > VOICE_THRESHOLD)
        {
            // The pre-trigger buffer already contains
            // the block that caused the trigger.
            startRecording();
        }
    }


    // ========================================================
    // CURRENTLY RECORDING
    // ========================================================

    else
    {
        addBlockToRecording();

        if (recordIndex >= AUDIO_SAMPLES)
        {
            finishRecording();
        }
    }
}


// ============================================================
// READ 20 ms FROM PDM MICROPHONE
// ============================================================

void readAudioBlock()
{
    size_t bytesNeeded = sizeof(blockBuffer);
    size_t bytesRead = 0;

    while (bytesRead < bytesNeeded)
    {
        size_t received = I2S.readBytes(
            ((char *)blockBuffer) + bytesRead,
            bytesNeeded - bytesRead
        );

        bytesRead += received;
    }
}


// ============================================================
// RMS VOICE DETECTION
// ============================================================

float calculateRMS(
    int16_t *samples,
    int count
)
{
    double sumSquares = 0.0;

    for (int i = 0; i < count; i++)
    {
        double sample = samples[i];

        sumSquares += sample * sample;
    }

    return sqrt(
        sumSquares / count
    );
}


// ============================================================
// START RECORDING
// ============================================================

void startRecording()
{
    recording = true;
    recordIndex = 0;

    digitalWrite(
        LIGHT_PIN,
        HIGH
    );
    copyPreTriggerToRecording();
    Serial.println("Voice detected");
}


// ============================================================
// COPY 20 ms BLOCK INTO 1-SECOND BUFFER
// ============================================================

void addBlockToRecording()
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        if (recordIndex < AUDIO_SAMPLES)
        {
            audioBuffer[recordIndex] =
                blockBuffer[i];

            recordIndex++;
        }
    }
}


// ============================================================
// RECORDING COMPLETE
// ============================================================

void finishRecording()
{
    recording = false;

    digitalWrite(
        LIGHT_PIN,
        LOW
    );

    Serial.println();
    Serial.println("===== RECORDING COMPLETE =====");

    Serial.print("Samples: ");
    Serial.println(recordIndex);

    Serial.print("Free heap before MFCC: ");
    Serial.println(ESP.getFreeHeap());

    Serial.println("Starting MFCC...");

    calculateMFCC(
        audioBuffer,
        mfcc
    );

    Serial.println("MFCC complete");

    Serial.print("Free heap after MFCC: ");
    Serial.println(ESP.getFreeHeap());

    Serial.println("Starting inference...");

    classifyVoice();

    Serial.println("Inference complete");

    Serial.print("Free heap after inference: ");
    Serial.println(ESP.getFreeHeap());

    Serial.println("==============================");
    Serial.println();

    recordIndex = 0;

    Serial.println("Ready for next recording.");
}

// ============================================================
// HZ -> MEL
// ============================================================

float hzToMel(float hz)
{
    return 2595.0f *
        log10f(
            1.0f + hz / 700.0f
        );
}


// ============================================================
// MEL -> HZ
// ============================================================

float melToHz(float mel)
{
    return 700.0f *
        (
            powf(
                10.0f,
                mel / 2595.0f
            )
            - 1.0f
        );
}


// ============================================================
// CREATE MEL FILTER LOCATIONS
// ============================================================

void createMelBins()
{
    float lowMel =
        hzToMel(0.0f);

    float highMel =
        hzToMel(
            SAMPLE_RATE / 2.0f
        );

    for (
        int i = 0;
        i < MEL_FILTERS + 2;
        i++
    )
    {
        float mel =
            lowMel +
            (
                (highMel - lowMel)
                * i
                / (MEL_FILTERS + 1)
            );

        float hz =
            melToHz(mel);

        melBins[i] =
            (int)floorf(
                (FFT_SIZE + 1)
                * hz
                / SAMPLE_RATE
            );
    }
}


// ============================================================
// HAMMING WINDOW
// ============================================================

void applyHammingWindow(
    float *frame
)
{
    for (
        int n = 0;
        n < FRAME_LENGTH;
        n++
    )
    {
        float window =
            0.54f
            -
            0.46f *
            cosf(
                2.0f
                * PI
                * n
                / (FRAME_LENGTH - 1)
            );

        frame[n] *= window;
    }
}


// ============================================================
// 512-POINT RADIX-2 FFT
// ============================================================

void runFFT(
    float *real,
    float *imag,
    int n
)
{
    // --------------------------------------------------------
    // Bit reversal
    // --------------------------------------------------------

    int j = 0;

    for (int i = 1; i < n; i++)
    {
        int bit = n >> 1;

        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j)
        {
            float temp = real[i];
            real[i] = real[j];
            real[j] = temp;

            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }
    }

    // --------------------------------------------------------
    // FFT stages
    // --------------------------------------------------------

    for (
        int length = 2;
        length <= n;
        length <<= 1
    )
    {
        float angle =
            -2.0f * PI / length;

        float wLengthReal =
            cosf(angle);

        float wLengthImag =
            sinf(angle);

        for (
            int i = 0;
            i < n;
            i += length
        )
        {
            float wReal = 1.0f;
            float wImag = 0.0f;

            for (
                int k = 0;
                k < length / 2;
                k++
            )
            {
                int even = i + k;

                int odd =
                    i + k
                    + length / 2;

                float oddReal =
                    real[odd] * wReal
                    -
                    imag[odd] * wImag;

                float oddImag =
                    real[odd] * wImag
                    +
                    imag[odd] * wReal;

                float evenReal =
                    real[even];

                float evenImag =
                    imag[even];

                real[even] =
                    evenReal + oddReal;

                imag[even] =
                    evenImag + oddImag;

                real[odd] =
                    evenReal - oddReal;

                imag[odd] =
                    evenImag - oddImag;

                float nextWReal =
                    wReal * wLengthReal
                    -
                    wImag * wLengthImag;

                float nextWImag =
                    wReal * wLengthImag
                    +
                    wImag * wLengthReal;

                wReal = nextWReal;
                wImag = nextWImag;
            }
        }
    }
}


// ============================================================
// POWER SPECTRUM
// Matches Python:
//
// power = magnitude^2 / FFT_SIZE
// ============================================================

void calculatePowerSpectrum(
    float *real,
    float *imag,
    float *power
)
{
    for (
        int k = 0;
        k <= FFT_SIZE / 2;
        k++
    )
    {
        power[k] =
            (
                real[k] * real[k]
                +
                imag[k] * imag[k]
            )
            /
            FFT_SIZE;
    }
}


// ============================================================
// MEL FILTER BANK + NATURAL LOG
// ============================================================

void calculateMelEnergies(
    float *power,
    float *logMel
)
{
    for (
        int m = 1;
        m <= MEL_FILTERS;
        m++
    )
    {
        int left =
            melBins[m - 1];

        int center =
            melBins[m];

        int right =
            melBins[m + 1];

        float energy = 0.0f;

        // Rising side
        for (
            int k = left;
            k < center;
            k++
        )
        {
            float denominator =
                center - left;

            if (denominator > 0)
            {
                float weight =
                    (float)(k - left)
                    / denominator;

                energy +=
                    power[k] * weight;
            }
        }

        // Falling side
        for (
            int k = center;
            k < right;
            k++
        )
        {
            float denominator =
                right - center;

            if (denominator > 0)
            {
                float weight =
                    (float)(right - k)
                    / denominator;

                energy +=
                    power[k] * weight;
            }
        }

        if (energy < 1e-10f)
        {
            energy = 1e-10f;
        }

        // Python used np.log()
        logMel[m - 1] =
            logf(energy);
    }
}


// ============================================================
// DCT -> MFCC
// ============================================================

void calculateDCT(
    float *logMel,
    float *mfccFrame
)
{
    for (
        int coefficient = 0;
        coefficient < MFCC_COEFFICIENTS;
        coefficient++
    )
    {
        float total = 0.0f;

        for (
            int m = 0;
            m < MEL_FILTERS;
            m++
        )
        {
            total +=
                logMel[m]
                *
                cosf(
                    PI
                    * coefficient
                    * (m + 0.5f)
                    / MEL_FILTERS
                );
        }

        mfccFrame[coefficient] =
            total;
    }
}


// ============================================================
// COMPLETE MFCC EXTRACTION
// ============================================================

void calculateMFCC(
    int16_t *audio,
    float outputMFCC[NUM_FRAMES][MFCC_COEFFICIENTS]
)
{
    for (
        int frameIndex = 0;
        frameIndex < NUM_FRAMES;
        frameIndex++
    )
    {
        int start =
            frameIndex * HOP_LENGTH;

        // Copy 400 samples
        for (
            int i = 0;
            i < FRAME_LENGTH;
            i++
        )
        {
            fftReal[i] =
                (float)audio[start + i];

            fftImag[i] = 0.0f;
        }

        // Zero pad 400 -> 512
        for (
            int i = FRAME_LENGTH;
            i < FFT_SIZE;
            i++
        )
        {
            fftReal[i] = 0.0f;
            fftImag[i] = 0.0f;
        }

        applyHammingWindow(
            fftReal
        );

        runFFT(
            fftReal,
            fftImag,
            FFT_SIZE
        );

        calculatePowerSpectrum(
            fftReal,
            fftImag,
            powerSpectrum
        );

        calculateMelEnergies(
            powerSpectrum,
            logMelBuffer
        );

        calculateDCT(
            logMelBuffer,
            outputMFCC[frameIndex]
        );

        // Give ESP32 system tasks some CPU time
        if ((frameIndex % 10) == 0)
        {
            yield();
        }
    }
}


// ============================================================
// ACTIVATION FUNCTIONS
// ============================================================

float relu(float x)
{
    if (x > 0.0f)
    {
        return x;
    }

    return 0.0f;
}


float sigmoid(float x)
{
    return 1.0f /
        (
            1.0f
            +
            expf(-x)
        );
}


// ============================================================
// FLATTEN 98 x 13 MFCC -> 1274 INPUTS
//
// Matches NumPy:
//
// mfcc.flatten()
// ============================================================

void flattenMFCC(
    float inputMFCC[NUM_FRAMES][MFCC_COEFFICIENTS]
)
{
    int index = 0;

    for (
        int frame = 0;
        frame < NUM_FRAMES;
        frame++
    )
    {
        for (
            int coefficient = 0;
            coefficient < MFCC_COEFFICIENTS;
            coefficient++
        )
        {
            modelInput[index] =
                inputMFCC[frame][coefficient];

            index++;
        }
    }
}


// ============================================================
// STANDARD SCALER
//
// Python:
//
// (X - mean) / scale
// ============================================================

void normalizeModelInput()
{
    for (
        int i = 0;
        i < MODEL_INPUT_SIZE;
        i++
    )
    {
        if (MODEL_SCALE[i] != 0.0f)
        {
            modelInput[i] =
                (
                    modelInput[i]
                    -
                    MODEL_MEAN[i]
                )
                /
                MODEL_SCALE[i];
        }
        else
        {
            modelInput[i] = 0.0f;
        }
    }
    Serial.println("First 10 normalized inputs:");

    for (int i = 0; i < 10; i++)
    {
        Serial.print(modelInput[i], 6);
        Serial.print(" ");
    }

    Serial.println();
}


// ============================================================
// HIDDEN DENSE LAYER
// ============================================================

void runHiddenLayer()
{
    for (
        int hidden = 0;
        hidden < MODEL_HIDDEN_SIZE;
        hidden++
    )
    {
        float sum =
            MODEL_B1[hidden];

        for (
            int input = 0;
            input < MODEL_INPUT_SIZE;
            input++
        )
        {
            int weightIndex =
                input
                * MODEL_HIDDEN_SIZE
                + hidden;

            sum +=
                modelInput[input]
                *
                MODEL_W1[weightIndex];
        }

        hiddenLayer[hidden] =
            relu(sum);
    }
}


// ============================================================
// BINARY OUTPUT LAYER
// ============================================================

float runOutputLayer()
{
    float sum =
        MODEL_B2[0];

    for (
        int hidden = 0;
        hidden < MODEL_HIDDEN_SIZE;
        hidden++
    )
    {
        sum +=
            hiddenLayer[hidden]
            *
            MODEL_W2[hidden];
    }

    return sigmoid(sum);
}


// ============================================================
// COMPLETE MODEL INFERENCE
// ============================================================

float runInference(
    float inputMFCC[NUM_FRAMES][MFCC_COEFFICIENTS]
)
{
    flattenMFCC(
        inputMFCC
    );

    normalizeModelInput();

    runHiddenLayer();

    return runOutputLayer();
}


// ============================================================
// YES / NO CLASSIFICATION
//
// Assumes:
//
// class 0 = YES
// class 1 = NO
//
// This corresponds to training with:
//
// CLASSES = ["yes", "no"]
// ============================================================

void classifyVoice()
{
    float probability =
        runInference(mfcc);

    Serial.print(
        "Model probability: "
    );

    Serial.println(
        probability,
        6
    );

    if (probability >= 0.5f)
    {
        digitalWrite(RED_LED, HIGH);
        digitalWrite(GREEN_LED, LOW);
        Serial.println(
            "Prediction: NO"
        );
    }
    else
    {
        digitalWrite(GREEN_LED, HIGH);
        digitalWrite(RED_LED, LOW);
        Serial.println(
            "Prediction: YES"
        );
    }
}

void updatePreTriggerBuffer()
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        preTriggerBuffer[preTriggerIndex] =
            blockBuffer[i];

        preTriggerIndex++;

        if (preTriggerIndex >= PRETRIGGER_SAMPLES)
        {
            preTriggerIndex = 0;
            preTriggerFull = true;
        }
    }
}

void copyPreTriggerToRecording()
{
    recordIndex = 0;

    // If the circular buffer has completely filled,
    // preTriggerIndex points to the oldest sample.
    if (preTriggerFull)
    {
        for (int i = 0; i < PRETRIGGER_SAMPLES; i++)
        {
            int sourceIndex =
                (preTriggerIndex + i)
                % PRETRIGGER_SAMPLES;

            audioBuffer[recordIndex] =
                preTriggerBuffer[sourceIndex];

            recordIndex++;
        }
    }

    // During startup, the buffer may not yet be full.
    else
    {
        for (int i = 0; i < preTriggerIndex; i++)
        {
            audioBuffer[recordIndex] =
                preTriggerBuffer[i];

            recordIndex++;
        }
    }
}