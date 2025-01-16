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

classNames = ['coke-bottle', 'coke-can', 'crisp', 'kitkat', 'lemon puff', 'pepsi-bottle', 'pepsi-can', 'unknown'] 
MIN_CONFIDENCE = 0.35  # Confidence threshold for known classes


def parse_price(price_str):
    # Normalize string: lowercase, remove currency symbols, and strip spaces
    price_str = price_str.lower().strip()
    
    if 'p' in price_str: 
        price_str = price_str.replace('p', '').strip()
        return float(price_str) / 100
    elif '£' in price_str:  
        price_str = price_str.replace('£', '').strip()
        return float(price_str)
    else:
        raise ValueError(f"Invalid price format: {price_str}")


def calculate_brightness(image, box):
    x1, y1, x2, y2 = box
    # region_of_interst
    roi = image[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv_roi[:, :, 2])
    return brightness

def draw_text_with_pillow(image, text, position, font_path="arial.ttf", font_size=32, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    bbox = draw.textbbox((0, 0), text, font=font)  # (left, top, right, bottom)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x,y = position
    text_bg_rect = [x, y, x + text_width + 10, y + text_height + 10]
    draw.rectangle(text_bg_rect, fill=bg_color)
    
    draw.text((x + 5, y), text, font=font, fill=text_color)
    #draw.text(position, text, font=font, fill=(0, 0, 255))  
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
            else: 
                #label = f"UNKNOWN {conf}"
                label = "UNKNOWN"

             # Calculate brightness of the bounding box area
            brightness = calculate_brightness(img, (x1, y1, x2, y2))
            if brightness > 128:  # Bright background
                text_color = (0, 0, 0)  # Dark text
                bg_color = (255, 255, 255)  # Light background
            else:  # Dark background
                text_color = (255, 255, 255)  # Light text
                bg_color = (0, 0, 0)  # Dark background

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255,0), 3)
            #cvzone.putTextRect(img, "£" , label, (max(10, x1), max(35, y1)))
            img = draw_text_with_pillow(img, label, (max(10, x1), max(35, y1)), text_color=text_color, bg_color=bg_color)

    total_price_label = f"Total: £{total_price:.2f}"
    img = draw_text_with_pillow(img, total_price_label, (10, 50),font_size=40, bg_color=(50, 50, 50))

    # Display the result
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
