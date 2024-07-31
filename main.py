import cv2
import pandas as pd
from datetime import datetime
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox

# Function to load student images and labels
def load_student_images(dataset_dir):
    images = []
    labels = []
    label_map = {}
    current_label = 0
    
    for student_name in os.listdir(dataset_dir):
        student_dir = os.path.join(dataset_dir, student_name)
        if os.path.isdir(student_dir):
            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(current_label)
            label_map[current_label] = student_name
            current_label += 1
    
    return images, labels, label_map

# Function to train the face recognizer
def train_face_recognizer(dataset_dir):
    images, labels, label_map = load_student_images(dataset_dir)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(images, np.array(labels))
    
    return face_recognizer, label_map

# Function to recognize faces and log attendance continuously
def recognize_faces_and_log_attendance(face_recognizer, label_map):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    # Set to track already recognized faces
    recognized_faces = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        recognized_this_frame = set()
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(roi_gray)
            
            if confidence < 100:
                name = label_map.get(label, "Unknown")
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if name not in recognized_faces:
                    log_attendance(name)
                    recognized_faces.add(name)
                    recognized_this_frame.add(name)
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        if len(recognized_this_frame) == 0:  # If no new faces were recognized
            print("No new faces recognized. Continuing...")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to log attendance
def log_attendance(name):
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    new_entry = pd.DataFrame({'Name': [name], 'Date': [current_date], 'Time': [current_time]})
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_path = 'logs/attendance_log.csv'
    if os.path.exists(log_path):
        attendance_log = pd.read_csv(log_path)
    else:
        attendance_log = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    
    if attendance_log[(attendance_log['Name'] == name) & (attendance_log['Date'] == current_date)].empty:
        attendance_log = pd.concat([attendance_log, new_entry], ignore_index=True)
        attendance_log.to_csv(log_path, index=False)
        print(f"Logged attendance for {name}.")
    else:
        print(f"Entry for {name} already exists for today.")

# Function to handle adding a new student
def add_new_student():
    student_name = simpledialog.askstring("Add New Student", "Enter the student's name:")
    if student_name:
        if not os.path.exists('datasets'):
            os.makedirs('datasets')
        student_dir = os.path.join('datasets', student_name)
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Add New Student', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                img_name = f"{student_name}_{count}.jpg"
                img_path = os.path.join(student_dir, img_name)
                cv2.imwrite(img_path, frame)
                print(f"Captured {img_name}")
                count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"New student {student_name} added successfully!")

# Function to handle taking attendance
def take_attendance():
    global face_recognizer, label_map
    print("Training face recognizer...")
    face_recognizer, label_map = train_face_recognizer('datasets')
    recognize_faces_and_log_attendance(face_recognizer, label_map)

# Function to create the main application window
def create_main_window():
    global face_recognizer, label_map
    # Create the main window
    root = tk.Tk()
    root.title("Student Attendance System")
    
    # Configure the main window
    root.geometry("800x500")  # Larger window
    root.configure(bg="#e0f7fa")  # Light cyan background color

    # Create a frame with a white background
    frame = tk.Frame(root, bg="#ffffff", padx=40, pady=30)
    frame.grid(sticky="nsew")

    # Add a title label
    title_label = tk.Label(frame, text="Student Attendance System", font=("Comic Sans MS", 28, "bold"), bg="#ffffff", fg="#00796b")
    title_label.grid(row=0, column=0, columnspan=3, pady=30)

    # Add buttons with custom style
    button_style = {
        "font": ("Comic Sans MS", 16),
        "bg": "#004d40",  # Dark teal
        "fg": "#ffffff",  # White text
        "bd": 0,
        "relief": "flat",
        "padx": 30,
        "pady": 15,
        "highlightbackground": "#004d40",
        "highlightcolor": "#004d40"
    }

    # Button widget creation and placement
    tk.Button(frame, text="Add New Student", command=add_new_student, **button_style).grid(row=1, column=0, pady=10, sticky="ew")
    tk.Button(frame, text="Take Attendance", command=take_attendance, **button_style).grid(row=2, column=0, pady=10, sticky="ew")
    tk.Button(frame, text="Exit", command=root.quit, **button_style).grid(row=3, column=0, pady=10, sticky="ew")

    # Configure row and column weights
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_rowconfigure(3, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    root.mainloop()

# Main function to start the application
if __name__ == "__main__":
    create_main_window()
