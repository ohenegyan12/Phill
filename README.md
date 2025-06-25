# 🎓 Student Performance Prediction System

An intelligent system that predicts student academic performance using advanced machine learning techniques. This application helps educators, students, and parents understand factors that influence academic success and provides actionable insights for improvement.

## ✨ Features

### 🎯 Core Functionality
- **Performance Prediction**: Input student information and receive accurate grade predictions
- **Interactive Web Interface**: User-friendly Streamlit application for easy data input and results visualization
- **Comprehensive Analysis**: Explore correlations between various factors and academic performance
- **Personalized Recommendations**: Get actionable insights based on prediction results

### 📊 Data Analysis
- **Feature Importance**: Understand which factors most influence student success
- **Correlation Analysis**: Explore relationships between different variables
- **Demographic Insights**: Analyze performance patterns across different student groups
- **Trend Visualization**: Interactive charts and graphs for data exploration

### 🎨 User Experience
- **Modern UI**: Beautiful, responsive interface with intuitive navigation
- **Real-time Predictions**: Instant results with detailed grade interpretations
- **Helpful Tooltips**: Contextual information for all input fields
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection (for initial setup)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd student-performance-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Launch the web application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📋 How to Use

### Making Predictions
1. Navigate to the "Predict Performance" section
2. Fill in the student information form:
   - **Academic Information**: Previous grades, study time, attendance
   - **Personal Factors**: Age, health status, family relationships
   - **Lifestyle Factors**: Free time activities, social interactions
3. Click "Predict Performance" to get results
4. Review the prediction and personalized recommendations

### Exploring Data Insights
1. Visit the "Data Insights" section
2. Explore various visualizations:
   - Grade distributions
   - Feature correlations
   - Demographic analysis
   - Academic performance patterns
3. Use the interactive tabs to focus on specific areas of interest

### Understanding Results
- **Grade Scale**: 0-20 scale (typical for Portuguese education system)
- **Grade Levels**: A (16-20), B (14-15), C (12-13), D (10-11), F (0-9)
- **Recommendations**: Personalized suggestions based on prediction results

## 📊 Factors Analyzed

### Academic Factors
- Previous period grades (G1, G2)
- Study time and learning habits
- Number of past failures
- School attendance and participation
- Extra educational support

### Personal Factors
- Age and maturity level
- Health status and well-being
- Family relationships and support
- Motivation and engagement levels

### Family Background
- Parental education levels
- Family size and structure
- Socioeconomic status
- Family support for education

### Lifestyle Factors
- Free time activities
- Social interactions and friendships
- Internet access and usage
- Health habits and behaviors

## 🎯 Prediction Target

The system predicts the **Final Grade (G3)** on a 0-20 scale, which represents:
- **0-9**: Failing grade (needs significant improvement)
- **10-11**: Below average (D grade)
- **12-13**: Average performance (C grade)
- **14-15**: Good performance (B grade)
- **16-20**: Excellent performance (A grade)

## 📁 Project Structure

```
student-performance-predictor/
├── app.py                      # Main web application
├── train_model.py              # Model training script
├── data_collection.py          # Data loading and generation
├── data_preprocessing.py       # Data cleaning and preparation
├── neural_network_model.py     # Neural network architecture
├── test_system.py              # System testing
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data storage
│   └── raw_student_data.csv
├── models/                     # Trained models
│   ├── *.h5                   # Neural network models
│   └── *.pkl                  # Data preprocessors
├── plots/                      # Generated visualizations
└── reports/                    # Training reports and results
```

## 🔧 Configuration

### Data Sources
The system supports multiple data sources:
- **UCI Student Performance Dataset**: Real educational data
- **Synthetic Data**: Generated data mimicking real patterns
- **Custom Datasets**: Your own student data (with proper formatting)

### Model Parameters
Key model settings can be adjusted in `neural_network_model.py`:
- Network architecture (layers, neurons)
- Training parameters (epochs, batch size)
- Regularization techniques
- Early stopping criteria

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_system.py
```

This will test:
- Data loading and preprocessing
- Model training and prediction
- Web application functionality
- Data visualization features

## 📈 Performance Metrics

The system provides several evaluation metrics:
- **Mean Squared Error (MSE)**: Overall prediction accuracy
- **R-squared Score**: Model fit quality
- **Feature Importance**: Which factors most influence predictions
- **Correlation Analysis**: Relationships between variables

## 🔒 Privacy & Ethics

### Data Privacy
- This system uses synthetic data for demonstration
- No real student information is collected or stored
- All predictions are based on anonymized patterns

### Ethical Considerations
- Educational AI should be used responsibly
- Consider potential biases in predictions
- Use predictions as guidance, not absolute truth
- Respect student privacy and confidentiality

### Fair Use Guidelines
- Use for educational and research purposes
- Avoid making decisions based solely on predictions
- Consider individual circumstances and context
- Maintain transparency about system limitations

## 🤝 Contributing

We welcome contributions to improve the system:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your improvements**
4. **Test thoroughly**
5. **Submit a pull request**

### Areas for Improvement
- Additional data sources
- Enhanced visualization features
- Improved model architectures
- Better user interface elements
- Additional evaluation metrics

## 📞 Support

### Common Issues
- **Model not found**: Run `python train_model.py` first
- **Import errors**: Ensure all dependencies are installed
- **Data loading issues**: Check data file format and location

### Getting Help
- Check the documentation in each module
- Review the test results for system status
- Ensure all dependencies are properly installed

## 📄 License

This project is designed for educational and research purposes. Please use responsibly and in accordance with applicable privacy and educational regulations.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the student performance dataset
- **Educational Research Community** for insights into academic prediction
- **Open Source Community** for the tools and libraries used

---

**Built with ❤️ for educational advancement**

*Empowering students, educators, and parents with data-driven insights* 