# Deployment Guide for Render

## Prerequisites

1. **Environment Variables**: Set these in your Render dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_API_KEY`: Your Google Search API key

2. **Dependencies**: All required packages are listed in `requirements.txt`

## Deployment Steps

1. **Push your code** to your Git repository
2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Create a new Web Service

3. **Configure the service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run agents/ui.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment**: Python 3.11

4. **Set environment variables** in the Render dashboard:
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`

5. **Deploy**: Click "Deploy" and wait for the build to complete

## Verification

Before deploying, run the verification script:

```bash
python deploy_check.py
```

This will check:
- All required imports work
- RAG components can be imported
- Environment variables are set

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are in `requirements.txt`
2. **Environment variables**: Verify they're set in Render dashboard
3. **Port issues**: The app uses `$PORT` environment variable automatically

### Build Failures

If the build fails:
1. Check the build logs in Render
2. Verify `requirements.txt` has all dependencies
3. Ensure Python version compatibility

## Local Testing

Test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"

# Run the app
streamlit run agents/ui.py
```

## File Structure

```
.
├── agents/
│   ├── ui.py          # Main Streamlit UI
│   ├── rag.py         # RAG system implementation
│   └── prompts/       # Prompt templates
├── documents/          # Document storage
├── requirements.txt    # Python dependencies
├── deploy_check.py    # Deployment verification
└── DEPLOYMENT.md      # This file
```
