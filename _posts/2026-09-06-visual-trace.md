---
layout: post
title: Visual Trace
description: Render Python functions as animated videos.
authors: [tlyleung]
x: 22
y: 58
---

Earlier in the year, I was working through a large collection of LeetCode problems. Although the professionally written solutions often helped, it was sometimes difficult to gain intuition. In some cases, I had to become a computer: hold an array in my head, walk two pointers along it, remember which numbers I had already seen and where. Often I would lose my train of thought and have to start over. So I would draw the row of boxes on a scrap of paper, cross out values, write new ones above them, and shuffle a pencil through the loop by hand. Given that the machine was already running through those exact steps anyway, it seemed like a strange thing to be doing by hand.

<figure>
  <img src="/assets/images/posts/visual-trace/two-sum.gif" alt="A traced run of two_sum: a highlight bar walks down the source listing on the left while a variables table on the right updates, with nums drawn as a row of cells and d as stacked key/value cells">
  <figcaption><code>two_sum.py</code> traced using Visual Trace.</figcaption>
</figure>

So I wrote [Visual Trace](https://github.com/tlyleung/visual-trace). It takes a Python function and renders it as a video using [Manim](https://www.manim.community), the animation engine Grant Sanderson built for [3Blue1Brown](https://www.youtube.com/@3blue1brown). A code highlight bar walks down the source code on the left, while a table of variables keeps pace on the right. Lists are drawn as rows of cells and dictionaries as stacked pairs. Nothing in the traced file refers to the library.

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    d = {}

    for i in range(len(nums)):
        num = nums[i]

        if target - num in d:
            return d[target - num], i
        else:
            d[num] = i


def main():
    args = ([2, 7, 11, 15], 9)
    func = two_sum
    return func, args
```

There is no import, `nums` is an ordinary list and `d` an ordinary `{}`. Only `main()` is used to hand over the function and the arguments to run it on. To trace the file, the library rewrites the syntax tree so that those literals become containers that pair the builtin with a Manim drawing of themselves, with every method overridden to update the visual representation as it goes along. A cell lights up because it's indexed, and the keys sweep because the code asks whether something was among them. Lists and dictionaries are finished and stacks, queues and trees are the obvious next things.

I built Visual Trace with Claude Code, and the first thing I had to do was close the verification loop. An agent cannot watch an MP4 and tell you that a frame looks wrong, so until the render produced something machine-readable there was no real way to iterate on it. The harness renders an example and then asserts on the geometry directly: where each cell sits in every settled frame, and what moved between one frame and the next. Anything geometry cannot express falls back to a captioned contact sheet, to be checked by a visual language model.