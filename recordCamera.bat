@echo off
set "OUTDIR=C:\recordings\cam1"
mkdir "%OUTDIR%" 2>nul

:loop
"C:\Users\551br\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe" ^
  -rtsp_transport tcp -i "rtsp://admin551:123456789@2.tcp.ngrok.io:15165/stream1" ^
  -c copy -f segment -segment_time 180 -reset_timestamps 1 -strftime 1 ^
  "%OUTDIR%\cam1_%%Y-%%m-%%d_%%H-%%M-%%S.mkv"

timeout /t 5 >nul
goto loop
