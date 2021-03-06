## Automatic Speech Recognition Feature Extraction Using OpenCL Improvements

Implemented using Intel OpenCL SDK.

### Prerequisites:  
	libsndfile
	FFTW(Pure CPU run for benchmarking an OpenCL run)
	Intel OpenCL SDK, with 2.1 support preferable. Run in x86 mode for Intel GPU support.


P.S: Benchmarking features have been planned, at least against an existing pure CPU based implementation using FFTW library for CPU based FFT. For running that, it will obviously become a prerequisite. Until you specifically want to use it, I'm trying to keep it free from that pre requisite.
  
#### Task Lists:
- [x] Code Added
- [x] Segmenting of soundfile
- [x] FFT using AppleFFT
- [x] Normalise and Filterbank
- [x] DCT
- [x] Delta and Delta-Delta
- [x] Soundfiles Added
- [x] Output Files Added
- [ ] Joining all modules for 'End-to-End'
- [ ] Add support Links for troubleshooting
- [ ] Add methodology documentation
- [ ] Add report and detailed documentation