using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Diagnostics;
using Whisper.net;

namespace SpeechToTextRealtime
{
    class Program
    {
        private static WhisperProcessor? _whisperProcessor;
        private static Process? _recordingProcess;
        private static MemoryStream _audioBuffer = new MemoryStream();
        private static readonly object _bufferLock = new object();
        private static CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();

        static void Main(string[] args)
        {
            Console.WriteLine("Inizializzazione Speech-to-Text in tempo reale...");

            try
            {
                // Inizializza Whisper
                InitializeWhisper();

                // Inizializza acquisizione audio
                InitializeAudio();

                // Avvia il processing in tempo reale
                _ = Task.Run(ProcessAudioContinuously, _cancellationTokenSource.Token);

                Console.WriteLine("Sistema attivo. Premi 'q' per uscire...");
                Console.WriteLine("Parla nel microfono per vedere la trascrizione.");

                // Loop principale
                while (true)
                {
                    var key = Console.ReadKey(true);
                    if (key.KeyChar == 'q' || key.KeyChar == 'Q')
                    {
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Errore: {ex.Message}");
            }
            finally
            {
                Cleanup();
            }
        }

        private static void InitializeWhisper()
        {
            Console.WriteLine("Caricamento modello Whisper tiny...");

            string modelPath = "models/ggml-small.bin";
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Modello non trovato in: {modelPath}");
                Console.WriteLine("Scarica il modello Whisper tiny da:");
                Console.WriteLine("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin");
                throw new FileNotFoundException("Modello Whisper non trovato");
            }

            var whisperFactory = WhisperFactory.FromPath(modelPath);
            _whisperProcessor = whisperFactory.CreateBuilder()
                .WithLanguage("it")
                .WithTemperature(0)
                .Build();

            Console.WriteLine("Modello Whisper caricato con successo.");
        }

        private static void InitializeAudio()
        {
            Console.WriteLine("Inizializzazione acquisizione audio...");

            // Per NetCoreAudio, dovrai implementare la cattura usando i tool nativi
            // oppure usare una libreria come CSCore (Windows) o altre alternative

            // Alternativa semplice: usa gli strumenti del sistema operativo
            string command, arguments;

            if (OperatingSystem.IsWindows())
            {
                // Windows: installa ffmpeg da https://ffmpeg.org/download.html
                command = "ffmpeg";
                arguments = "-f dshow -i audio=\"Microphone\" -ar 16000 -ac 1 -f s16le pipe:1";
            }
            else if (OperatingSystem.IsMacOS())
            {
                // macOS: brew install sox
                command = "sox";
                arguments = "-d -r 16000 -c 1 -b 16 -e signed-integer -t raw -";
            }
            else
            {
                // Linux: pre-installato in molte distro
                command = "arecord";
                arguments = "-f S16_LE -r 16000 -c 1";
            }

            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = command,
                    Arguments = arguments,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                _recordingProcess = new Process { StartInfo = startInfo };
                _recordingProcess.OutputDataReceived += OnAudioDataReceived;
                _recordingProcess.ErrorDataReceived += OnErrorReceived;

                _recordingProcess.Start();
                _recordingProcess.BeginOutputReadLine();
                _recordingProcess.BeginErrorReadLine();

                Console.WriteLine($"Acquisizione audio avviata usando: {command}");
            }
            catch (Exception ex)
            {
                throw new Exception($"Errore inizializzazione audio. Installa {command}: {ex.Message}");
            }
        }

        private static void OnAudioDataReceived(object sender, DataReceivedEventArgs e)
        {
            if (e.Data != null && e.Data.Length > 0)
            {
                // Converte la stringa in byte array (assumendo raw audio data)
                byte[] audioBytes = System.Text.Encoding.Latin1.GetBytes(e.Data);

                lock (_bufferLock)
                {
                    _audioBuffer.Write(audioBytes, 0, audioBytes.Length);
                }
            }
        }

        private static void OnErrorReceived(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                Console.WriteLine($"Audio error: {e.Data}");
            }
        }

        private static async Task ProcessAudioContinuously()
        {
            const int chunkSizeBytes = 16000 * 2 * 5;  // 5 secondi
            const int stepSizeBytes = 16000 * 2 * 3;   // Avanza di 3 secondi
            byte[] processingBuffer = new byte[chunkSizeBytes];

            while (!_cancellationTokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    bool hasAudio = false;

                    lock (_bufferLock)
                    {
                        if (_audioBuffer.Length >= chunkSizeBytes)
                        {
                            _audioBuffer.Position = 0;
                            _audioBuffer.Read(processingBuffer, 0, chunkSizeBytes);

                            byte[] remaining = new byte[_audioBuffer.Length - stepSizeBytes];
                            _audioBuffer.Position = stepSizeBytes;
                            _audioBuffer.Read(remaining, 0, remaining.Length);

                            _audioBuffer.SetLength(0);
                            _audioBuffer.Write(remaining, 0, remaining.Length);

                            hasAudio = true;
                        }
                    }

                    if (hasAudio)
                    {
                        await ProcessAudioChunk(processingBuffer);
                    }

                    await Task.Delay(500, _cancellationTokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
        }

        private static async Task ProcessAudioChunk(byte[] audioData)
        {
            try
            {
                float[] audioSamples = ConvertBytesToFloats(audioData);

                if (!HasSufficientAudioEnergy(audioSamples))
                    return;

                await foreach (var segment in _whisperProcessor!.ProcessAsync(audioSamples))
                {
                    if (!string.IsNullOrWhiteSpace(segment.Text))
                    {
                        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {segment.Text.Trim()}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Errore trascrizione: {ex.Message}");
            }
        }

        private static float[] ConvertBytesToFloats(byte[] audioBytes)
        {
            float[] floats = new float[audioBytes.Length / 2];

            for (int i = 0; i < floats.Length; i++)
            {
                short sample = BitConverter.ToInt16(audioBytes, i * 2);
                floats[i] = sample / 32768.0f;
            }

            return NormalizeAudio(floats);
        }

        private static float[] NormalizeAudio(float[] samples)
        {
            float max = samples.Max(Math.Abs);
            if (max > 0.1f)
            {
                float factor = 0.8f / max;
                for (int i = 0; i < samples.Length; i++)
                    samples[i] *= factor;
            }
            return samples;
        }

        private static bool HasSufficientAudioEnergy(float[] samples)
        {
            const float energyThreshold = 0.001f;

            double energy = 0;
            foreach (var sample in samples)
            {
                energy += sample * sample;
            }

            double rms = Math.Sqrt(energy / samples.Length);
            return rms > energyThreshold;
        }

        private static void Cleanup()
        {
            Console.WriteLine("\nChiusura in corso...");

            _cancellationTokenSource.Cancel();

            if (_recordingProcess != null && !_recordingProcess.HasExited)
            {
                try
                {
                    _recordingProcess.Kill();
                    _recordingProcess.WaitForExit(2000);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Errore chiusura processo: {ex.Message}");
                }
                finally
                {
                    _recordingProcess?.Dispose();
                }
            }

            _whisperProcessor?.Dispose();
            _audioBuffer?.Dispose();
            _cancellationTokenSource?.Dispose();

            Console.WriteLine("Risorse liberate.");
        }
    }
}