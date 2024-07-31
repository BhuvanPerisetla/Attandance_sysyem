import os
import cv2
import argparse

# Function to capture and save images of a new student
def add_new_student(student_name):
    print(f"Adding new student: {student_name}")
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

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add a new student.')
    parser.add_argument('student_name', type=str, help='The name of the new student')
    args = parser.parse_args()
    
    add_new_student(args.student_name)
