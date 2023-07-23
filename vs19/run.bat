:: win batch code for simplicity, program can also work over main code directly
@ECHO OFF
:: Default parameters:
SET height=832
SET width=832

:: This batch file use for standart using of program with default params
:: also this file is video search area
cd /D .\x64\Debug 
IF ERRORLEVEL 1 (ECHO cd failed) ELSE (ECHO path changed successfully...)

ECHO please choose one of input data which you want to use:

FOR /F %%G in ('dir /B *.webm *.mp4 *.mkv *.flv *.avi') do (
%SystemRoot%\System32\choice.exe /C YN /N /M "Is %%G taken as an input[Y/N]?" 
if errorlevel 2 (echo skipping option %%G) else CALL:startcode "%%G")
goto : EOF

:startcode
ECHO %~1 is processing as video data ...
.\Detect_and_Track.exe --config=yolov4_car.cfg --model=yolov4_car.weights --classes=coco.names --height=%height% --width=%width% --scale=0.00392 --rgb --input=.\%~1
pause
EXIT
goto : EOF
