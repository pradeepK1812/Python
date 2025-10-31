from weather import get_weather
from colorama import init, Fore, Style

# Initialize colorama (auto-reset colors after print)
init(autoreset=True)

def color_for_temp(temp):
    if temp < 10:
        return Fore.CYAN  # Cold
    elif temp < 25:
        return Fore.GREEN  # Pleasant
    elif temp < 35:
        return Fore.YELLOW  # Warm
    else:
        return Fore.RED  # Hot


def main():
    print(Fore.CYAN + Style.BRIGHT + "\n🌦  Welcome to the Weather CLI Tool!\n")
    city = input(Fore.WHITE + "Enter city name: ").strip()
    
    try:
        data = get_weather(city)
        temp = data["temperature"]
        wind = data["windspeed"]
        weather = data["weather"]
        resolved_city = data["resolved_city"]

        color = color_for_temp(temp)
        print(
            f"\n{Style.BRIGHT}{Fore.MAGENTA}📍 Location:{Style.RESET_ALL} {resolved_city}"
        )
        print(
            f"{Style.BRIGHT}{Fore.BLUE}🌡  Temperature:{Style.RESET_ALL} {color}{temp}°C"
        )
        print(
            f"{Style.BRIGHT}{Fore.CYAN}💨 Wind Speed:{Style.RESET_ALL} {wind} km/h"
        )
        print(
            f"{Style.BRIGHT}{Fore.YELLOW}☁️  Condition:{Style.RESET_ALL} {weather}\n"
        )

    except Exception as e:
        print(Fore.RED + f"❌ Error: {e}")

if __name__ == "__main__":
    main()
