from routes import route

def individual_serial_user(usr) -> dict:
    return {
        "user_id" : str(usr["_id"]),
        "firstname" : usr["firstname"],
        "surname" : usr["surname"],
        "email" : usr["email"],
        "username" : usr["username"],
        "password" : usr["password"],
        "is_admin" : usr["is_admin"],
        "is_verified" : usr["is_verified"],
        "user_verification_file_path" : usr["user_verification_file_path"]
    }

def individual_serial_factory(facto) -> dict:
    return {
        "factory_id" : str(facto["_id"]),
        "factory_name" : facto["factory_name"],
        "factory_details" : facto["factory_details"],
        "is_disable" : facto["is_disable"]
    }

def individual_serial_build(build) -> dict:
    return {
        "building_id" : str(build["_id"]),
        "building_name" : build["building_name"],
        "building_length" : build["building_length"],
        "building_width" : build["building_width"],
        "building_latitude" : build["building_latitude"],
        "building_longitude" : build["building_longitude"],
        "data_location" : build["data_location"],
        "factory_id" : str(build["factory_id"])
    }

def individual_serial_history(histo) -> dict:
    return {
        "history_id" : str(histo["_id"]),
        "create_date" : histo["create_date"],
        "create_time" : histo["create_time"],
        "is_process" : histo["is_process"],
        "history_path" : histo["history_path"],
        "building_id" : str(histo["building_id"])
    }

def individual_serial_image(img) -> dict:
    return{
        "image_id" : str(img["_id"]),
        "image_path" : img["image_path"],
        "x_index" : img["x_index"],
        "y_index" : img["y_index"],
        "is_user_verified" : img["is_user_verified"],
        "history_id" : str(img["history_id"])
    }

def individual_serial_defectlo(defectlo) -> dict:
    return{
        "defect_location_id" : str(defectlo["_id"]),
        "class_type" : defectlo["class_type"],
        "x" : defectlo["x"],
        "y" : defectlo["y"],
        "w" : defectlo["w"],
        "h" : defectlo["h"],
        "image_id" : str(defectlo["image_id"]),
        "class_name" : str(defectlo["class_name"]),
        "is_user_verified" : defectlo["is_user_verified"]
    }

def individual_serial_defec(defec) -> dict:
    return{
        "defect_class" : defec["defect_class"],
        "defect_class_name" : defec["defect_class_name"]
    }

def individual_serial_permis(permis) -> dict:
    return{
        "user_id" : str(permis["user_id"]),
        "factory_id" : str(permis["factory_id"]),
    }

def individual_serial_log(log) -> dict:
    return {
        "actor" : str(log["actor"]),
        "message" : str(log["message"]),
        "timestamp" : str(log["timestamp"]),
    }

def list_serial_user(usr_lis) -> list :
    return [individual_serial_user(usr) for usr in usr_lis]

def list_serial_factory(facto_lis) -> list :
    return [individual_serial_factory(facto) for facto in facto_lis]

def list_serial_build(build_lis) -> list :
    return [individual_serial_build(build) for build in build_lis]

def list_serial_histo(histo_lis) -> list :
    return [individual_serial_history(histo) for histo in histo_lis]

def list_serial_image(img_lis) -> list :
    return [individual_serial_image(img) for img in img_lis]

def list_serial_defectlo(defectlo_lis) -> list :
    return [individual_serial_defectlo(defectlo) for defectlo in defectlo_lis]

def list_serial_defec(defec_lis) -> list :
    return [individual_serial_defec(defec) for defec in defec_lis]

def list_serial_permis(permis_list) -> list :
    return [individual_serial_permis(permis) for permis in permis_list]

def list_serial_log(log_list) -> list :
    return [individual_serial_log(log) for log in log_list]


