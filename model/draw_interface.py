import pygame
import sys
import numpy as np
import os
import json
from ogham_convnext import REPO_ROOT, predict_image

pygame.init()

WINDOW_SIZE = 800
DRAWING_SIZE = 400
CANVAS_COLOR = (255, 255, 255)
DRAWING_COLOR = (0, 0, 0)
LINE_WIDTH = 5

BUTTON_WIDTH = 200
BUTTON_HEIGHT = 40
BUTTON_COLOR = (50, 150, 50)
BUTTON_HOVER_COLOR = (70, 170, 70)
BUTTON_TEXT_COLOR = (255, 255, 255)

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Ogham Character Drawing")

drawing_surface = pygame.Surface((DRAWING_SIZE, DRAWING_SIZE))
drawing_surface.fill(CANVAS_COLOR)

def draw_button(surface, text, rect, color):
    pygame.draw.rect(surface, color, rect, border_radius=5)
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, BUTTON_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=rect.center)
    surface.blit(text_surface, text_rect)

def is_mouse_over_button(pos, button_rect):
    return button_rect.collidepoint(pos)

def save_drawing():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    temp_path = os.path.join(output_dir, "drawn_character.png")
    pygame.image.save(drawing_surface, temp_path)
    return temp_path

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "models" / "convnext_adapted.pt")
    
    drawing = False
    last_pos = None
    prediction_text = "Draw an Ogham character"
    confidence_text = ""
    
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    predict_button = pygame.Rect(
        (WINDOW_SIZE - BUTTON_WIDTH) // 2,
        WINDOW_SIZE - 100,
        BUTTON_WIDTH,
        BUTTON_HEIGHT
    )
    clear_button = pygame.Rect(
        (WINDOW_SIZE - BUTTON_WIDTH) // 2,
        WINDOW_SIZE - 50,
        BUTTON_WIDTH,
        BUTTON_HEIGHT
    )

    while True:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if is_mouse_over_button(mouse_pos, predict_button):
                        image_path = save_drawing()
                        try:
                            result = predict_image(model_path, image_path)
                            prediction = result['predicted']
                            confidence = result['confidence']
                            top_predictions = result['top_predictions']
                            
                            prediction_text = f"Predicted: {prediction}"
                            confidence_text = f"Confidence: {confidence*100:.1f}%"
                            
                            confidence_text += "\nTop 3: " + " | ".join([f"{letter} ({conf*100:.1f}%)" for letter, conf in top_predictions])
                        except Exception as e:
                            prediction_text = f"Error: {str(e)}"
                            confidence_text = ""
                    elif is_mouse_over_button(mouse_pos, clear_button):
                        drawing_surface.fill(CANVAS_COLOR)
                        prediction_text = "Draw an Ogham character"
                        confidence_text = ""
                    else:
                        drawing = True
                        last_pos = event.pos
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click released
                    drawing = False
                    
            elif event.type == pygame.MOUSEMOTION and drawing:
                if last_pos:
                    current_pos = event.pos
                    start_x = (last_pos[0] - (WINDOW_SIZE - DRAWING_SIZE) // 2)
                    start_y = (last_pos[1] - (WINDOW_SIZE - DRAWING_SIZE) // 2)
                    end_x = (current_pos[0] - (WINDOW_SIZE - DRAWING_SIZE) // 2)
                    end_y = (current_pos[1] - (WINDOW_SIZE - DRAWING_SIZE) // 2)
                    
                    pygame.draw.line(drawing_surface, DRAWING_COLOR,
                                  (start_x, start_y), (end_x, end_y), LINE_WIDTH)
                    last_pos = current_pos
        screen.fill((200, 200, 200))
        
        drawing_rect = drawing_surface.get_rect(
            center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        screen.blit(drawing_surface, drawing_rect)
        
        text = font.render(prediction_text, True, (0, 0, 0))
        text_rect = text.get_rect(center=(WINDOW_SIZE // 2, 50))
        screen.blit(text, text_rect)
        
        if confidence_text:
            y_offset = 80
            for line in confidence_text.split('\n'):
                conf_text = small_font.render(line, True, (0, 0, 0))
                conf_rect = conf_text.get_rect(center=(WINDOW_SIZE // 2, y_offset))
                screen.blit(conf_text, conf_rect)
                y_offset += 25
        
        predict_color = BUTTON_HOVER_COLOR if is_mouse_over_button(mouse_pos, predict_button) else BUTTON_COLOR
        clear_color = BUTTON_HOVER_COLOR if is_mouse_over_button(mouse_pos, clear_button) else BUTTON_COLOR
        
        draw_button(screen, "Predict", predict_button, predict_color)
        draw_button(screen, "Clear", clear_button, clear_color)
        
        pygame.display.flip()

if __name__ == "__main__":
    main()
