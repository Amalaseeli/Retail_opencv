from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

model = YOLO('best_model.pt')  # Load the YOLO model
prices = {
    'coke-bottle' : '£1.25',
    'coke-can' : '£1.05',
    'crisp' : '£1.25',
    'kitkat': '85p',
    'lemon puff' : '75p',
    'pepsi-bottle' : '£1.09',
    'pepsi-can' : '£1.00'
}

classNames = ['coke-bottle', 'coke-can', 'crisp', 'kitkat', 'lemon puff', 'pepsi-bottle', 'pepsi-can', 'unknown']  # Known classes
MIN_CONFIDENCE = 0.35  # Confidence threshold for known classes


def parse_price(price_str):
    # Normalize string: lowercase, remove currency symbols, and strip spaces
    price_str = price_str.lower().strip()
    
    if 'p' in price_str:  # Handle prices in pence (e.g., '75p')
        price_str = price_str.replace('p', '').strip()
        return float(price_str) / 100
    elif '£' in price_str:  # Handle prices in pounds (e.g., '£1.25')
        price_str = price_str.replace('£', '').strip()
        return float(price_str)
    else:
        raise ValueError(f"Invalid price format: {price_str}")


def draw_text_with_pillow(image, text, position, font_path="arial.ttf", font_size=32):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(0, 0, 255))  
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


while True:
    sucess, img =cap.read()
    results= model(img, stream=True)

    total_price = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence and Class
            conf = math.ceil((box.conf[0]) * 100) / 100
            cls = int(box.cls[0]) if conf > MIN_CONFIDENCE and int(box.cls[0]) < len(classNames) else None

            # Determine label
            if cls is not None:  # Known object
                #label = f"{classNames[cls]} {conf}"
                className = classNames[cls]
                price_str = prices.get(className, "N/A")
                price = parse_price(price_str) if price_str != "N/A" else 0.0
                total_price += price
                label = f"{className} {price_str}"
                print(label)
            else:  # Unknown object
                #label = f"UNKNOWN {conf}"
                label = "UNKNOWN"

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255,0), 3)
            #cvzone.putTextRect(img, "£" , label, (max(10, x1), max(35, y1)))
            img = draw_text_with_pillow(img, label, (max(10, x1), max(35, y1)))

    total_price_label = f"Total: £{total_price:.2f}"
    img = draw_text_with_pillow(img, total_price_label, (10, 50))

    # Display the result
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
