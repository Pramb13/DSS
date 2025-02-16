from setuptools import setup, find_packages

setup(
    name="driver-drowsiness-detection",  # Name of your app/package
    version="0.1",
    packages=find_packages(),  # Automatically find packages
    install_requires=[  # List of dependencies
        "streamlit",
        "opencv-python",
        "numpy",
        "mediapipe",
        "playsound",  # Optional
    ],
    entry_points={  # Optional: if you want to add command-line tools
        'console_scripts': [
            'start-drowsiness-detection = app:main',  # Replace 'main' with the actual function name in app.py to run
        ],
    },
    classifiers=[  # Optional classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
