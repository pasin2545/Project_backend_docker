from pydantic import BaseModel,Field
from typing import List

class User(BaseModel):
    firstname : str
    surname : str
    email :str
    username : str = Field(max_length=20)
    password : str
    is_admin : bool = False
    is_verified : bool = False
    user_verification_file_path : str

class Factory(BaseModel):
    factory_name : str
    factory_details : str
    is_disable : bool = False

class Building(BaseModel):
    building_name : str
    building_length : float
    building_width : float
    building_latitude : str
    building_longitude : str
    data_location : str
    factory_id : str

class CreateBuildingRequest(BaseModel):
    building_name : str
    building_length : float
    building_width : float
    building_latitude : str
    building_longitude :str
    factory_id : str

class Image(BaseModel):
    image_path :str
    x_index : int
    y_index : int
    history_id : str
    is_user_verified : bool = False

class DefectLocation(BaseModel):
    class_type : int
    x : float
    y : float
    w : float
    h : float
    is_user_verified : bool = False

class Defect(BaseModel):
    defect_class : int
    defect_class_name : str

class Permission(BaseModel):
    username : str
    factory_name : str
    factory_details : str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class CreateUserRequest(BaseModel):
    firstname : str
    surname : str
    email : str
    username: str
    password : str
    verified_file_path: str

class CreateAdminRequest(BaseModel):
    firstname : str
    surname : str
    email : str
    username : str
    password : str
    verified_file_path :str

class ExtractVideo(BaseModel):
    input_dir: str
    output_dir: str

class VerifiedUser(BaseModel):
    verified : bool
    username : str

class UserChangePassword(BaseModel):
    username : str
    old_password : str
    new_password :str

class ChangeRole(BaseModel):
    username : str
    user_isadmin : bool = False

class AdminChangePassword(BaseModel):
    username : str
    new_password : str

class UsernameInput(BaseModel):
    username : str

class FactoryId(BaseModel):
    facto_id : str

class BuildingId(BaseModel):
    build_id : str

class HistoryPath(BaseModel):
    history_path : str

class ImagePath(BaseModel):
    image_path : str

class ImageId(BaseModel):
    image_id : str

class DefectLocationWithImage(BaseModel):
    defectlos : List[DefectLocation]
    Image_post_id : str

class BuildingDetail(BaseModel):
    building_id : str
    building_length : float
    building_width : float
    building_latitude : str
    building_longitude :str

class History(BaseModel):
    create_date : str
    create_time : str
    is_process : bool = False
    history_path : str
    building_id : str

class HistoryId(BaseModel):
    histo_id : str

class UserFac(BaseModel):
    user_id : str
    fac_id : str