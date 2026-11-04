# Submission summary

Full name: Hudiya e Zehra
Roll number: i230761
Class / section: Section B
University email: [i230761@isb.nu.edu.pk](mailto:i230761@isb.nu.edu.pk)
GitHub username: Hudiyahaha
Agent name: StudyPlan Agent
Domain: Study-Plan Builder

GitHub repository URL: [https://github.com/Hudiyahaha/StudyPlan-Agent](https://github.com/Hudiyahaha/StudyPlan-Agent)
Final commit hash: PENDING (demo branch commit, added after the push)
Working agent interface: PENDING (Render URL)
Health endpoint (GET): PENDING (Render URL)/health
Arena endpoint (POST): PENDING (Render URL)/arena/run
Manifest endpoint (GET): PENDING (Render URL)/arena/manifest
API documentation: PENDING (Render URL)/docs

Hosting provider: Render (free plan)
Default model / provider: qwen/qwen3.8-27b / Groq
Other available models: openai/gpt-oss-120b, openai/gpt-oss-20b (also used as automatic failover)
Example input: I have PDC assignment due Tuesday needing 4 hours and AI quiz Wednesday needing 3 hours. Classes Monday 8:30-14:30 and Tuesday 10:00-13:00. Make a study plan.
Expected result: A completed, conflict-free plan that places all 7 hours outside class times and before each deadline.
Cold-start / restart limitations: The free Render service sleeps when idle, so the first request can take about a minute. Chat history is held in memory and is lost on restart or sleep. The deployment uses one worker.
Repository access: PENDING (instructor invited / access confirmed)
Public test results: evaluation/public_results.json

## Update after the original submission

This summary describes a version completed after the original Google Classroom submission, for demonstration. It moves the agent to a live Groq model, replaces the earlier model experiment, and adds the public deployment. The earlier experiment results were invalid: the xAI account had no credits, so those runs fell back to the offline mock. Commits show their actual dates.
