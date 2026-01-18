"""
PC Controller for Eye Control
"""
import pyautogui
import time
from typing import Dict, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PCController:
    """Control PC using eye tracking intents"""
    
    def __init__(self, 
                 screen_width: int = None,
                 screen_height: int = None,
                 sensitivity: float = 0.7,
                 smoothing: float = 0.3):
        """
        Initialize PC controller
        
        Args:
            screen_width: Screen width (auto-detect if None)
            screen_height: Screen height (auto-detect if None)
            sensitivity: Mouse movement sensitivity
            smoothing: Cursor movement smoothing factor
        """
        # Get screen size
        if screen_width is None or screen_height is None:
            self.screen_width, self.screen_height = pyautogui.size()
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height
        
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        
        # State
        self.last_cursor_pos = pyautogui.position()
        self.enabled = False
        self.dwell_start_time = None
        self.dwell_threshold = 1.5  # seconds
        
        # Safety settings
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.01
        
        logger.info(f"PCController initialized (screen: {self.screen_width}x{self.screen_height})")
    
    def execute_intent(self, intent: str, features: Dict = None):
        """
        Execute control action based on intent
        
        Args:
            intent: Intent action ('click', 'scroll_up', 'scroll_down', 'move', 'idle')
            features: Eye tracking features
        """
        if not self.enabled:
            return
        
        try:
            if intent == 'click':
                self.click()
            elif intent == 'scroll_up':
                self.scroll(direction='up')
            elif intent == 'scroll_down':
                self.scroll(direction='down')
            elif intent == 'move' and features:
                self.move_cursor(features)
            elif intent == 'idle':
                pass  # Do nothing
            
        except Exception as e:
            logger.error(f"Error executing intent '{intent}': {e}")
    
    def move_cursor(self, features: Dict):
        """
        Move cursor based on gaze direction
        
        Args:
            features: Eye tracking features
        """
        gaze_direction = features.get('gaze_direction', 'center')
        
        # Get current position
        current_x, current_y = pyautogui.position()
        
        # Calculate movement
        move_amount = int(20 * self.sensitivity)
        
        new_x, new_y = current_x, current_y
        
        if gaze_direction == 'left':
            new_x = max(0, current_x - move_amount)
        elif gaze_direction == 'right':
            new_x = min(self.screen_width - 1, current_x + move_amount)
        elif gaze_direction == 'up':
            new_y = max(0, current_y - move_amount)
        elif gaze_direction == 'down':
            new_y = min(self.screen_height - 1, current_y + move_amount)
        
        # Apply smoothing
        smooth_x = int(current_x + (new_x - current_x) * self.smoothing)
        smooth_y = int(current_y + (new_y - current_y) * self.smoothing)
        
        # Move cursor
        if (smooth_x, smooth_y) != (current_x, current_y):
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)
            logger.debug(f"Cursor moved to ({smooth_x}, {smooth_y})")
    
    def click(self, button: str = 'left'):
        """
        Perform mouse click
        
        Args:
            button: Mouse button ('left', 'right', 'middle')
        """
        pyautogui.click(button=button)
        logger.info(f"{button.capitalize()} click performed")
    
    def double_click(self):
        """Perform double click"""
        pyautogui.doubleClick()
        logger.info("Double click performed")
    
    def scroll(self, direction: str = 'up', amount: int = 3):
        """
        Scroll mouse wheel
        
        Args:
            direction: Scroll direction ('up' or 'down')
            amount: Scroll amount
        """
        scroll_amount = amount if direction == 'up' else -amount
        pyautogui.scroll(scroll_amount)
        logger.debug(f"Scrolled {direction} by {amount}")
    
    def type_text(self, text: str):
        """
        Type text
        
        Args:
            text: Text to type
        """
        pyautogui.write(text, interval=0.05)
        logger.info(f"Typed text: {text}")
    
    def press_key(self, key: str):
        """
        Press keyboard key
        
        Args:
            key: Key name (e.g., 'enter', 'space', 'esc')
        """
        pyautogui.press(key)
        logger.info(f"Pressed key: {key}")
    
    def enable(self):
        """Enable PC control"""
        self.enabled = True
        logger.info("PC control enabled")
    
    def disable(self):
        """Disable PC control"""
        self.enabled = False
        logger.info("PC control disabled")
    
    def is_enabled(self) -> bool:
        """Check if PC control is enabled"""
        return self.enabled
    
    def set_sensitivity(self, sensitivity: float):
        """Set cursor movement sensitivity"""
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        logger.info(f"Sensitivity set to {self.sensitivity}")
    
    def set_smoothing(self, smoothing: float):
        """Set cursor movement smoothing"""
        self.smoothing = max(0.0, min(1.0, smoothing))
        logger.info(f"Smoothing set to {self.smoothing}")
