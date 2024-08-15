import carla


def list_maps():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    available_maps = client.get_available_maps()
    print("Available maps:")
    for map_name in available_maps:
        print(map_name)


def change_map(map_name):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world(map_name)
    print(f"Map changed to {map_name}")


if __name__ == "__main__":
    # List available maps
    list_maps()

    # Change to the desired map
    change_map('/Game/Carla/Maps/Town01')
