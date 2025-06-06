
---

## Quick Notes

### Date

`2025-06-02`

### Key Points

* Definition of efficiency
* Manual inspection and stat for dark count
* 2-fold efficiency vs 1-fold

---

### Definition of efficiency
* Setup
    Individual channel fixed OF  
    ch1-44 use uv template extracted from TraceSimulator  
    ch45 uses ch45 qp template  
    [ch46:] use ch46+ qp template  
* Definition of passrate(efficiency)
    For 1fold definition: as long as one channel has ampl>threshold then we label the event as passed
    For 2fold definition: the second largest ampl>threshold then the event labeled as passed.
---

### Manual Inspection & Dark Count Statistics
* Setup
    [Same as Definition of efficiency section](#Definition of efficiency)
* Observations
    Some noise traces visually resemble signal templates with amplitudes around 1. Their χ² values (~0.87) tend to be lower than signal fits (typically 0.88–1.0).
    For submerged channels, 2-fold triggering does not significantly improve AUC over 1-fold. However, multi-fold triggers might be promising for vacuum channels or NR events.
* Stat of noise fitting amplitudes
    Stat has been made with 1000 set of MMC noise traces with noise power of 33.2.Below is the histogram for ampl fitted for three type channels. Shows their distribution rough the same. Thus decide to use overall one threshold trigger.
    <img src="image.png" alt="distribution of ampl for 3 different channels" width="500"/>
* Due to event-wise triggering, the effective rejection threshold is higher than the average noise amplitude  
    <img src="image-2.png" alt="passrate plot" width="500"/>


---
### Study on the 2-fold
To assess if a 2-fold trigger improves performance, ROC curves for both 1-fold and 2-fold were plotted:
1. AUC Comparison: The 2-fold trigger shows a smaller AUC than the 1-fold, indicating reduced overall performance.
2. Check the threshold value at FPR==0.05. The value of thresholds indicate: 
    i. Threshold Value: The 1-fold threshold is significantly higher than the 2-fold, as expected, since 2-fold compares the second-largest amplitude.

    ii. True Positive Rate (TPR): At the same FPR, 1-fold achieves a higher TPR. This suggests the 2-fold threshold fails to reject noise effectively, and many valid single-channel UV signals are missed, lowering TPR.

3. ER vs NR ROC: The ROC curves for ER and NR events are nearly identical, confirming that the NR template is well defined.
4. Same Threshold Behavior: At the same threshold, the 2-fold trigger yields both lower TPR and FPR, consistent with expectations.
5. TPR at Low FPR: At low FPR, TPR is higher for ER events than NR, suggesting the UV-based trigger more efficiently collects signal—likely because UV signals deposite into one channel, whereas QPs spread over multiple channels.  
    <img src="image-1.png" alt="ROC curve plot for 1 and 2 fold" width="500"/>

### AUC as a function of recoil energy for True samples
1. Setup: Fixed OF amplitude fitting with channel-wise templates. Samples with energy from 5 to 50, with NR and ER mixed(100+100 per energy point). Haven't observed evidence 2fold has better performance than one.   
<img src="image-3.png" alt="AUC vs energy" width="500"/>
2. 2Fold may able to efficiently reject the fake signal, but by comparison at same TPR level, haven't observed evidence.  
<img src="image-4.png" alt="TPR vs energy at FPR==0.01" width="500"/>
---

### Noise Spectrum → Noise PSD (TODO)

Plan: Derive a conversion factor to match the theoretical noise spectrum with observed PSD magnitudes from MMC noise used in the paper.

---

### Question
* For the noise traces with amplitude, they usually have chisq ~0.87 and is smaller than the chisq for fitting signal traces, could this be used for classifier?  
* multi-fold might be useful for vac channel, should I spend time study whether and how much?
* What further studies would be most helpful? Prioritization guidance?