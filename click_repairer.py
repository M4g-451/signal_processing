"""
stages/click_repairer.py
------------------------
Repairs corrupted samples found by ClickDetector.

Two strategies based on how long the corrupt region is:
  - Short (8 samples or fewer): linear interpolation between clean boundaries
  - Longer: fill with the median of nearby clean samples, with a short
    fade at each edge to avoid introducing new clicks at the boundary
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ClickRepairer:
    """
    Args:
        short_threshold: Corrupt regions up to this length use interpolation
        context_radius:  How many samples on each side to use for median fill
        fade_length:     Crossfade length at the edges of medium-length repairs
    """

    def __init__(self, short_threshold=8, context_radius=128, fade_length=16):
        self.short_threshold = short_threshold
        self.context_radius  = context_radius
        self.fade_length     = fade_length

    def repair(self, signal, click_mask):
        """
        Repair all corrupted regions in the signal.

        Returns the repaired signal and a summary dict with repair counts.
        """
        if signal.shape != click_mask.shape:
            raise ValueError(f"Shape mismatch: signal {signal.shape} vs mask {click_mask.shape}")

        repaired = signal.copy()
        regions  = self._find_regions(click_mask)

        short_count  = 0
        medium_count = 0

        for start, end in regions:
            if (end - start) <= self.short_threshold:
                self._repair_short(repaired, start, end)
                short_count += 1
            else:
                self._repair_medium(repaired, start, end)
                medium_count += 1

        total = short_count + medium_count
        logger.info("Repaired %d region(s): %d short, %d medium", total, short_count, medium_count)

        return repaired, {"total": total, "short": short_count, "medium": medium_count}

    def _repair_short(self, signal, start, end):
        """Linear interpolation between the last clean sample before and after."""
        n = len(signal)
        left  = signal[start - 1] if start > 0 else signal[end]
        right = signal[end] if end < n else signal[start - 1]

        length = end - start
        for i, idx in enumerate(range(start, end)):
            t = (i + 1) / (length + 1)
            signal[idx] = left + t * (right - left)

    def _repair_medium(self, signal, start, end):
        """Fill with local median, fade in/out at the edges."""
        n = len(signal)

        # Gather clean samples from either side
        left_ctx  = signal[max(0, start - self.context_radius) : start]
        right_ctx = signal[end : min(n, end + self.context_radius)]
        context   = np.concatenate([left_ctx, right_ctx])

        fill = float(np.median(context)) if len(context) > 0 else 0.0
        signal[start:end] = fill

        # Smooth the boundary with a short Hann fade
        fade_len = min(self.fade_length, (end - start) // 2)
        if fade_len > 1:
            fade_in  = np.hanning(fade_len * 2)[:fade_len]
            fade_out = np.hanning(fade_len * 2)[fade_len:]

            if start > 0:
                region = slice(start, start + fade_len)
                signal[region] = signal[start - 1] * (1 - fade_in) + fill * fade_in

            if end < n:
                region = slice(end - fade_len, end)
                signal[region] = fill * fade_out + signal[end] * (1 - fade_out)

    @staticmethod
    def _find_regions(mask):
        """Convert a boolean mask to a list of (start, end) tuples."""
        regions = []
        in_region = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False
        if in_region:
            regions.append((start, len(mask)))
        return regions
