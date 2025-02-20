# Interactive Simulation Learning Platform

An interactive learning platform for teaching simulation concepts through a real-world food truck case study. Built with Streamlit and Python, this platform helps students understand the complete simulation process from random number generation to statistical analysis.

## 🌟 Features

### 📖 Story Mode
- Interactive food truck business case presentation
- Visual representation of service stations and workflow
- Real-world context for simulation concepts

### 🎲 Random Number Generation
- Linear Congruential Generator (LCG) demonstration
- Linear Feedback Shift Register (LFSR) implementation
- Various sampling methods (Inverse Transform, Box-Muller, etc.)
- Interactive visualization of random number generation

### 📊 Distribution Fitting
- Statistical distribution analysis
- Goodness-of-fit tests
- Parameter estimation and confidence intervals
- Interactive distribution comparison tools

### 🔄 Simulation Engine
- Event-based simulation implementation
- Real-time system state visualization
- Queue and service time monitoring
- Performance metrics tracking

### 📈 Alternative Analysis
- Statistical comparison of system alternatives
- Confidence interval calculations
- Sample size determination
- Decision support visualization

## 🛠️ Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **NumPy & SciPy**: Statistical computations



## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simulation-learning-platform.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run story.py
```

## 📁 Project Structure

```
simulation-learning-platform/
├── story.py                    # Main story and system definition
├── random_generator.py         # Random number generation module
├── goodness_of_fit.py         # Distribution fitting module
├── compare_alternatives.py     # Alternative analysis module
├── event_simulation.py        # Core simulation engine
├── show_simulation_steps.py   # Step-by-step simulation display
└── requirements.txt           # Project dependencies
```

## 💡 Usage

The platform is organized into five interactive modules that guide students through the simulation process:

1. **Story Module**: Introduces the food truck case study and system parameters
2. **Random Number Generation**: Demonstrates various methods of generating random numbers
3. **Distribution Fitting**: Shows how to analyze and fit statistical distributions
4. **Event Simulation**: Provides real-time visualization of the simulation process
5. **Alternative Analysis**: Enables statistical comparison of system alternatives

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- Developed as part of the Simulation Course at the Department of Industrial Engineering and Management, Ben-Gurion University.
- Special thanks to the Department of Teaching Innovation
