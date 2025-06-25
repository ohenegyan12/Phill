#!/bin/bash

echo "🚀 Setting up Student Performance Predictor for deployment..."

# Create necessary directories
mkdir -p models plots reports data

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Train the model if it doesn't exist
if [ ! -f "models/student_performance_model_*.h5" ]; then
    echo "🤖 Training the model..."
    python train_model.py
fi

# Test the system
echo "🧪 Testing the system..."
python test_system.py

echo "✅ Setup complete! Your application is ready for deployment."
echo ""
echo "🎯 Recommended deployment options:"
echo "1. Streamlit Cloud: https://share.streamlit.io"
echo "2. Heroku: heroku create && git push heroku main"
echo "3. Railway: https://railway.app"
echo "4. Render: https://render.com"
echo ""
echo "📖 See deploy_to_netlify.md for detailed instructions." 