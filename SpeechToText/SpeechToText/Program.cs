using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using Whisper.net;


namespace SpeechToTextRealtime
{
    class Program
    {
        private static WhisperProcessor? _whisperProcessor;
        private static WaveInEvent? _waveIn;
        private static MemoryStream _audioBuffer = new MemoryStream();
        private static readonly object _bufferLock = new object();
        private static CancellationTokenSource _cancellationTokenSource = new CancellationTokenSource();

        static async Task Main(string[] args)
        {
            Console.WriteLine("Inizializzazione Speech-to-Text in tempo reale...");
            
            try
            {
                // Inizializza Whisper
                await InitializeWhisper();
                
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

        private static async Task InitializeWhisper()
        {
            Console.WriteLine("Caricamento modello Whisper tiny...");
            
            // Verifica se il modello esiste
            string modelPath = "models/ggml-tiny.bin";
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Modello non trovato in: {modelPath}");
                Console.WriteLine("Scarica il modello Whisper tiny da:");
                Console.WriteLine("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin");
                throw new FileNotFoundException("Modello Whisper non trovato");
            }

            var whisperFactory = WhisperFactory.FromPath(modelPath);
            _whisperProcessor = whisperFactory.CreateBuilder()
                .WithLanguage("it") // Italiano
                .WithThreads(Environment.ProcessorCount)
                //.WithSpeedup(true)
                .Build();
                
            Console.WriteLine("Modello Whisper caricato con successo.");
        }

        private static void InitializeAudio()
        {
            Console.WriteLine("Inizializzazione acquisizione audio...");
            
            _waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(16000, 16, 1), // 16kHz, 16-bit, mono
                BufferMilliseconds = 100 // Buffer di 100ms
            };

            _waveIn.DataAvailable += OnAudioDataAvailable;
            _waveIn.RecordingStopped += OnRecordingStopped;
            
            _waveIn.StartRecording();
            Console.WriteLine("Acquisizione audio avviata.");
        }

        private static void OnAudioDataAvailable(object? sender, WaveInEventArgs e)
        {
            lock (_bufferLock)
            {
                _audioBuffer.Write(e.Buffer, 0, e.BytesRecorded);
            }
        }

        private static void OnRecordingStopped(object? sender, StoppedEventArgs e)
        {
            if (e.Exception != null)
            {
                Console.WriteLine($"Errore acquisizione audio: {e.Exception.Message}");
            }
        }

        private static async Task ProcessAudioContinuously()
        {
            const int chunkSizeBytes = 16000 * 2; // 1 secondo di audio (16kHz * 2 bytes per sample)
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
                            int bytesRead = _audioBuffer.Read(processingBuffer, 0, chunkSizeBytes);
                            
                            // Rimuovi i dati processati dal buffer
                            byte[] remaining = new byte[_audioBuffer.Length - bytesRead];
                            _audioBuffer.Read(remaining, 0, remaining.Length);
                            _audioBuffer.SetLength(0);
                            _audioBuffer.Write(remaining, 0, remaining.Length);
                            
                            hasAudio = bytesRead == chunkSizeBytes;
                        }
                    }

                    if (hasAudio)
                    {
                        await ProcessAudioChunk(processingBuffer);
                    }
                    
                    await Task.Delay(50, _cancellationTokenSource.Token); // 50ms di pausa
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Errore processing audio: {ex.Message}");
                    await Task.Delay(1000, _cancellationTokenSource.Token);
                }
            }
        }

        private static async Task ProcessAudioChunk(byte[] audioData)
        {
            try
            {
                // Converti byte[] in float[] per Whisper
                float[] audioSamples = ConvertBytesToFloats(audioData);
                
                // Verifica se c'è abbastanza energia audio (evita processing del silenzio)
                if (!HasSufficientAudioEnergy(audioSamples))
                    return;

                // Process con Whisper
                await foreach (var segment in _whisperProcessor.ProcessAsync(audioSamples))
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
            float[] floats = new float[audioBytes.Length / 2]; // 16-bit = 2 bytes per sample
            
            for (int i = 0; i < floats.Length; i++)
            {
                short sample = BitConverter.ToInt16(audioBytes, i * 2);
                floats[i] = sample / 32768.0f; // Normalizza a [-1.0, 1.0]
            }
            
            return floats;
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
            
            _waveIn?.StopRecording();
            _waveIn?.Dispose();
            
            _whisperProcessor?.Dispose();
            _audioBuffer?.Dispose();
            
            Console.WriteLine("Risorse liberate.");
        }
    }

    // Classe di supporto per Sherpa ONNX (implementazione semplificata)
    public class SherpaOnnxRecognizer
    {
        public static SherpaOnnxRecognizer Create(string modelPath)
        {
            // Implementazione placeholder per Sherpa ONNX
            // In una implementazione reale, qui caricheresti il modello ONNX
            return new SherpaOnnxRecognizer();
        }

        public string Recognize(float[] audioSamples)
        {
            // Placeholder - implementa la logica di riconoscimento Sherpa ONNX
            return string.Empty;
        }

        public void Dispose()
        {
            // Cleanup risorse Sherpa ONNX
        }
    }
}