import carla

def list_vehicle_blueprints():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    for blueprint in vehicle_blueprints:
        print(blueprint.id)

list_vehicle_blueprints()
