using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenTK.Audio.OpenAL;
using System.Runtime.InteropServices;

namespace SpeechToTextConsole
{
    class Program
    {
        private static InferenceSession? _session;
        private static ALCaptureDevice _captureDevice;
        private static List<short> _audioBuffer = new List<short>();
        private static readonly object _bufferLock = new object();
        private static bool _isRecording = false;
        private static Thread? _captureThread;

        static async Task Main(string[] args)
        {
            Console.WriteLine("Speech to Text - .NET Core + ONNX (Cross-Platform)");
            Console.WriteLine("====================================================");

            // Rileva la piattaforma
            DetectPlatform();

            // Inizializza il modello ONNX
            InitializeOnnxModel("whisper-tiny.onnx");

            // Configura cattura audio cross-platform
            SetupAudioCapture();

            Console.WriteLine("Premi SPAZIO per iniziare/fermare la registrazione, ESC per uscire");

            bool isRecording = false;

            while (true)
            {
                var key = Console.ReadKey(true).Key;

                if (key == ConsoleKey.Escape)
                    break;

                if (key == ConsoleKey.Spacebar)
                {
                    if (!isRecording)
                    {
                        StartRecording();
                        Console.WriteLine("🎤 Registrazione avviata...");
                        isRecording = true;
                    }
                    else
                    {
                        StopRecording();
                        Console.WriteLine("⏹️ Registrazione fermata. Elaborazione...");

                        // Processa l'audio catturato
                        var text = await ProcessAudioBuffer();
                        Console.WriteLine($"Testo: {text}");
                        Console.WriteLine();

                        isRecording = false;
                    }
                }
            }

            Cleanup();
        }

        static void DetectPlatform()
        {
            string platform = "Sconosciuto";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                platform = "Windows";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                platform = "macOS";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                platform = "Linux";

            Console.WriteLine($"🖥️  Piattaforma rilevata: {platform}");
        }

        static void InitializeOnnxModel(string modelPath)
        {
            try
            {
                var sessionOptions = new SessionOptions
                {
                    EnableCpuMemArena = false,
                    EnableMemoryPattern = false
                };

                _session = new InferenceSession(modelPath, sessionOptions);
                Console.WriteLine($"✅ Modello ONNX caricato: {modelPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore nel caricare il modello: {ex.Message}");
                Console.WriteLine("Scarica whisper-tiny.onnx da: https://github.com/openai/whisper");
                Environment.Exit(1);
            }
        }

        static void SetupAudioCapture()
        {
            try
            {
                // Lista dispositivi di input disponibili
                var captureDevices = ALC.GetStringList(GetEnumerationStringList.CaptureDeviceSpecifier).ToList();

                if (captureDevices?.Count() > 0)
                {
                    Console.WriteLine($"🎤 Dispositivo audio: {captureDevices[0]}");

                    // Apri il dispositivo di cattura (16kHz, mono, 16-bit)
                    _captureDevice = ALC.CaptureOpenDevice(captureDevices[0], 16000, ALFormat.Mono16, 4096);

                    if (_captureDevice != IntPtr.Zero)
                    {
                        Console.WriteLine("✅ Cattura audio configurata (16kHz mono, cross-platform)");
                    }
                    else
                    {
                        throw new Exception("Impossibile aprire il dispositivo di cattura");
                    }
                }
                else
                {
                    throw new Exception("Nessun dispositivo di input trovato");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Errore configurazione audio: {ex.Message}");
                Environment.Exit(1);
            }
        }

        static void StartRecording()
        {
            lock (_bufferLock)
            {
                _audioBuffer.Clear();
            }

            _isRecording = true;
            ALC.CaptureStart(_captureDevice);

            // Avvia thread di cattura
            _captureThread = new Thread(CaptureAudioLoop)
            {
                IsBackground = true
            };
            _captureThread.Start();
        }

        static void StopRecording()
        {
            _isRecording = false;
            ALC.CaptureStop(_captureDevice);
            _captureThread?.Join(1000); // Aspetta max 1 secondo
        }

        static void CaptureAudioLoop()
        {
            const int bufferSize = 1024;
            short[] buffer = new short[bufferSize];

            while (_isRecording)
            {
                // CORREZIONE: Usa ALC.GetInteger con AlcGetInteger.CaptureSamples
                int samplesAvailable = ALC.GetInteger(_captureDevice, AlcGetInteger.CaptureSamples);

                if (samplesAvailable > 0)
                {
                    int samplesToRead = Math.Min(samplesAvailable, bufferSize);
                    ALC.CaptureSamples(_captureDevice, buffer, samplesToRead);

                    lock (_bufferLock)
                    {
                        for (int i = 0; i < samplesToRead; i++)
                        {
                            _audioBuffer.Add(buffer[i]);
                        }
                    }
                }

                Thread.Sleep(10); // Evita un loop troppo veloce
            }
        }

        static async Task<string> ProcessAudioBuffer()
        {
            try
            {
                float[] audioData;

                lock (_bufferLock)
                {
                    var samples = _audioBuffer.ToArray();
                    audioData = ConvertSamplesToFloat(samples);
                }

                if (audioData.Length == 0)
                    return "[Nessun audio catturato]";

                // Prepara input per Whisper
                var inputTensor = new DenseTensor<float>(audioData, new[] { 1, audioData.Length });
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("audio", inputTensor)
                };

                // Esegui inferenza
                using var results = await Task.Run(() => _session!.Run(inputs));

                // Estrai il testo dall'output
                var outputTensor = results.FirstOrDefault()?.AsTensor<long>();
                if (outputTensor != null)
                {
                    return DecodeTokens(outputTensor.ToArray());
                }

                return "[Errore nell'elaborazione]";
            }
            catch (Exception ex)
            {
                return $"[Errore: {ex.Message}]";
            }
        }

        static float[] ConvertSamplesToFloat(short[] samples)
        {
            var floats = new float[samples.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                floats[i] = samples[i] / 32768f; // Normalizza a [-1, 1]
            }
            return floats;
        }

        static string DecodeTokens(long[] tokens)
        {
            // Implementazione semplificata del decoder
            // In una versione completa, dovresti usare il tokenizer di Whisper
            var text = string.Join("", tokens.Select(t => ((char)t).ToString()));
            return text.Trim();
        }

        static void Cleanup()
        {
            _isRecording = false;
            _captureThread?.Join(1000);

            if (_captureDevice != IntPtr.Zero)
            {
                ALC.CaptureCloseDevice(_captureDevice);
            }

            _session?.Dispose();
            Console.WriteLine("Risorse rilasciate.");
        }
    }
}