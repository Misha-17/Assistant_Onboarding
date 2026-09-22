# Fixed local code: image path and language are data supplied through process-local
# environment variables. Recognized document text is never evaluated as code.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStreamWithContentType, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$limitMs = [int]$env:SISU_OCR_TIMEOUT_MS
if ($limitMs -lt 1 -or $limitMs -gt 60000) { throw 'Invalid OCR timeout' }
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and
    $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Await-OcrOperation($operation, [Type]$resultType) {
    $task = $asTask.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $remaining = $limitMs - [int]$timer.ElapsedMilliseconds
    if ($remaining -le 0 -or -not $task.Wait($remaining)) { throw 'OCR operation timeout' }
    return $task.Result
}
$language = New-Object Windows.Globalization.Language($env:SISU_OCR_LANGUAGE)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) { throw 'Requested OCR language unavailable' }
$file = Await-OcrOperation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($env:SISU_OCR_IMAGE_PATH)) ([Windows.Storage.StorageFile])
$stream = Await-OcrOperation ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
try {
    $decoder = Await-OcrOperation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    if ($decoder.PixelWidth -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or
        $decoder.PixelHeight -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension) { throw 'Image dimensions exceed OCR engine limit' }
    $bitmap = Await-OcrOperation ($decoder.GetSoftwareBitmapAsync([Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8, [Windows.Graphics.Imaging.BitmapAlphaMode]::Ignore)) ([Windows.Graphics.Imaging.SoftwareBitmap])
    try {
        $recognized = Await-OcrOperation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $lines = @()
        foreach ($line in $recognized.Lines) {
            $words = @()
            foreach ($word in $line.Words) {
                $rectangle = $word.BoundingRect
                $words += @{text=$word.Text; box=@([double]$rectangle.X, [double]$rectangle.Y,
                          [double]$rectangle.Width, [double]$rectangle.Height); confidence=$null}
            }
            $lines += @{text=$line.Text; words=$words; confidence=$null}
        }
        $output = @{schema_version=1; backend='windows-media-ocr'; backend_version=[Environment]::OSVersion.Version.ToString();
                    language=$engine.RecognizerLanguage.LanguageTag; image_width=[int]$decoder.PixelWidth;
                    image_height=[int]$decoder.PixelHeight; text_angle=$recognized.TextAngle;
                    lines=$lines; confidence=$null; elapsed_s=$timer.Elapsed.TotalSeconds}
        $output | ConvertTo-Json -Compress -Depth 12
    } finally { if ($null -ne $bitmap) { $bitmap.Dispose() } }
} finally { if ($null -ne $stream) { $stream.Dispose() } }
