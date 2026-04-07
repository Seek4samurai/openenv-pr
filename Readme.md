# Content Moderation using ML + OpenEnv

## Problem

Detect NSFW / Violence content in images/videos.

## Solution

- Built a custom environment using OpenEnv
- Used CNN models for:
  - NSFW detection
  - Violence detection
- Combined predictions to classify content

## How it works

1. Input image/frame
2. Model predicts scores
3. Environment assigns label
4. Agent takes action

## Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app

```

## Run using docker

```bash
docker build -t service .
docker run -p 8000:8000 service

```

## API Dashboard

Head to `http://localhost:8000/docs/` on your browser. And You'll see 3 APIs `/reset`, `/step`, and `/state`
