@echo off
set "OUTDIR=C:\recordings\cam1"
mkdir "%OUTDIR%" 2>nul

rem Set RTSP_URL as an environment variable before running this script,
rem e.g.: set RTSP_URL=rtsp://user:pass@host:554/stream1
if not defined RTSP_URL (
  echo RTSP_URL environment variable is not set.
  exit /b 1
)

:loop
"C:\Users\551br\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe" ^
  -rtsp_transport tcp -i "%RTSP_URL%" ^
  -c copy -f segment -segment_time 180 -reset_timestamps 1 -strftime 1 ^
  "%OUTDIR%\cam1_%%Y-%%m-%%d_%%H-%%M-%%S.mkv"

timeout /t 5 >nul
goto loop
