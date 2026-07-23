# Recording a GitHub demo GIF

A short demo GIF makes this project easier to understand at a glance. Record the app only after the trained model files are in place and the webcam is working.

## Recommended content

Keep the recording between 8 and 12 seconds:

1. Start on the running Gesture2Voice app.
2. Show one recognised gesture or fingerspelled letter.
3. Let the predicted text appear on screen.
4. Click **Speak** so viewers can see the complete workflow.

Avoid recording any personal information, browser tabs, notification pop-ups, or the camera feed of anyone who has not agreed to be included.

## Windows workflow

1. Install and open [ScreenToGif](https://www.screentogif.com/).
2. Choose **Recorder**, select only the browser window containing Gesture2Voice, and record at 10–15 frames per second.
3. Trim the beginning and end so the result starts immediately.
4. Export the recordings as `gesture-to-word.gif` and `fingerspelling-to-word.gif`.
5. Save both files to `docs/assets/`.
6. Replace the temporary placeholder image lines in the main README under **Demos** with:

```markdown
![Gesture to word demo](docs/assets/gesture-to-word.gif)
![Fingerspelling to word demo](docs/assets/fingerspelling-to-word.gif)
```

Aim to keep the final GIF under 10 MB. For larger recordings, upload an MP4 to a GitHub Release and link to it from the README instead.
