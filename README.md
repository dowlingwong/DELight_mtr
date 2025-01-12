# CV_HEX_QC
## Introduction
This is the repo for HGCAl baseplate quality control at KIT in 2024.
More detailed documentation and hardware list could be found [here](QC_short_doc.pdf). If links in the pdf cannot be opened on browser, please try after download. 


## wk1
- [x] ~~Stitching~~
    -[x]~~use features toallocate and rescale~~
- [ ] ~~Stereo vision measurement~~
    - [ ] ~~mechanical holder for 2 cams~~
- [x] features to locate and rescale
    - [x] image processing 
    - [x] gray scale
    - [x] non-local means denoising
    - [x] binarize(threshold cut-off)
- [x] contour detection
- [x] focus, aperture code
- [x] line detection
    - [x] hough
- [x] circle detection 
    - [x] hough

---
## wk2
- [ ] Upgrade all dependencies to latest stable/LTS, adjust syntax accordingly
- [ ] create a base docker image and singularity environment
- [ ] deploy exercise data to helix
