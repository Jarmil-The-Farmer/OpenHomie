import pygame
import numpy as np
import time

class JoystickController:
    def __init__(self, deadzone=0.1, max_velocity=1.0):
        """
        Inicializace joysticku.
        :param deadzone: Pásmo necitlivosti (aby robot neujížděl sám od sebe).
        :param max_velocity: Maximální rychlost v m/s pro škálování.
        """
        pygame.init()
        pygame.joystick.init()
        
        self.deadzone = deadzone
        self.max_velocity = max_velocity
        self.joystick = None

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"🎮 Připojen joystick: {self.joystick.get_name()}")
        else:
            print("⚠️ Žádný joystick nenalezen! Vracím nulové hodnoty.")

    def _apply_deadzone(self, value):
        """Ořízne malé hodnoty šumu."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_command(self):
        """
        Čte stav joysticku.
        :return: tuple (velocity_vector, height)
                 - velocity_vector: np.array [v_x, v_y, 0.0]
                 - height: float 0.0 až 1.0
        """
        if not self.joystick:
            return np.array([0.0, 0.0, 0.0]), 0.5  # Defaultní hodnoty při chybě

        # Je nutné zavolat pump, aby pygame načetl nové eventy
        pygame.event.pump()

        # --- LEVÁ PÁČKA (Pohyb) ---
        # Axis 1 je obvykle vertikální osa levé páčky (vpřed/vzad)
        # Axis 0 je horizontální osa levé páčky (vlevo/vpravo)
        # Poznámka: Na Linuxu může být Y osa invertovaná (nahoru je -1), proto dáváme minus.
        raw_x = self.joystick.get_axis(1) * -1  # Dopředu/Dozadu (Osa X robota)
        raw_y = self.joystick.get_axis(0) * -1  # Vlevo/Vpravo (Osa Y robota)
        raw_z = self.joystick.get_axis(3) * -1
        
        vx = self._apply_deadzone(raw_x) * self.max_velocity
        vy = self._apply_deadzone(raw_y) * self.max_velocity
        vz = self._apply_deadzone(raw_z) * self.max_velocity

        
        # Sestavení 3D vektoru (Z složka je 0, protože páčka je 2D)
        velocity_vector = np.array([vx, vy, vz], dtype=np.float32)

        # --- PRAVÁ PÁČKA (Výška) ---
        # Axis 4 je obvykle vertikální osa pravé páčky na Xbox ovladačích (na PS4 to může být Axis 3)
        # Hodnota je od -1 (nahoru) do 1 (dolů).
        # Chceme mapovat: Nahoru (-1) -> 1.0 (vysoko), Dolů (1) -> 0.0 (nízko).
        raw_height = self.joystick.get_axis(4) 
        
        # Vzorec: (-val + 1) / 2  =>  (-(-1)+1)/2 = 1.0  ...  (-(1)+1)/2 = 0.0
        height = (-raw_height + 2) / 2.0
        
        # Oříznutí pro jistotu, aby to nebylo < 0 nebo > 1
        height = np.clip(height, 0.4, 0.8)

        return velocity_vector, height

# --- TESTOVACÍ SMYČKA ---
if __name__ == "__main__":
    # Tuto část spusť pro otestování, jestli joystick reaguje správně
    controller = JoystickController(max_velocity=2.0)
    
    print("Testování joysticku... (Ctrl+C pro ukončení)")
    try:
        while True:
            vel, h = controller.get_command()
            
            # Formátovaný výpis pro kontrolu
            print(f"\rRychlost [X, Y, Z]: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] | Výška: {h:.2f}", end="")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest ukončen.")