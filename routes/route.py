# coding: utf-8
from fastapi import APIRouter, Depends, HTTPException, Request, File, UploadFile
from fastapi.responses import FileResponse
from models.model import (
    User,
    Factory,
    Building,
    Image,
    Defect,
    DefectLocation,
    Permission,
    Token,
    TokenData,
    CreateUserRequest,
    ExtractVideo,
    VerifiedUser,
    UserChangePassword,
    ChangeRole,
    AdminChangePassword,
    UsernameInput,
    FactoryId,
    BuildingId,
    HistoryPath,
    ImagePath,
    ImageId,
    DefectLocationWithImage,
    BuildingDetail,
    CreateAdminRequest,
    History,
    HistoryId,
    CreateBuildingRequest,
    UserFac,
)
from config.database import (
    collection_user,
    collection_building,
    collection_factory,
    collection_Image,
    collection_DefectLocation,
    collection_Defect,
    collection_Permission,
    collection_history,
    collection_log,
)
from schema.schemas import (
    list_serial_user,
    list_serial_build,
    list_serial_factory,
    list_serial_image,
    list_serial_defectlo,
    list_serial_defec,
    list_serial_permis,
    list_serial_histo,
    list_serial_log,
)
from datetime import datetime
import pytz
from bson import ObjectId
from typing import List, Annotated, Optional
import asyncio
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from datetime import timedelta, datetime
from pydantic import BaseModel
from starlette import status
from config.database import db
import shutil
import uuid
import ultralytics
import torch
import torchvision
from ultralytics import YOLO
import os
import cv2
from ultralytics.utils.plotting import Annotator
import numpy as np
import time
import json
import glob
import re
import math
from GPSPhoto import gpsphoto
import smtplib
from email.message import EmailMessage
import ssl

router = APIRouter()

SECRET_KEY = "Roof_Surface"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def getCurTime():
    current_datetime_utc = datetime.utcnow()
    timezone_bangkok = pytz.timezone("Asia/Bangkok")
    current_datetime_bangkok = current_datetime_utc.replace(tzinfo=pytz.utc).astimezone(
        timezone_bangkok
    )
    return current_datetime_bangkok


# -------------------------------------------------------Auth-------------------------------------------------------


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_user(user: User):
    hashed_password = pwd_context.hash(user.password)
    user.password = hashed_password
    collection_user.insert_one(user.dict())


def get_user(username: str):
    user_data = collection_user.find_one({"username": username})
    if user_data:
        return User(**user_data)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    user_verified = collection_user.find_one({"username": username})
    if not user or not verify_password(password, user.password):
        return False
    if user_verified["is_verified"] == False:
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


@router.post("/upload_user_file")
async def upload_file(file: UploadFile):
    file_path = str(uuid.uuid4())
    os.makedirs(f"/app/data/user_file_verified/{file_path}", exist_ok=True)
    file_dir = f"/app/data/user_file_verified/{file_path}/{file.filename}"
    try:
        contents = await file.read()
        with open(file_dir, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"There was an error uploading the file. {e}"
        )
    finally:
        await file.close()

    return {"path": file_dir}


@router.post("/sign_up", status_code=status.HTTP_201_CREATED)
async def sign_up(create_user_request: CreateUserRequest):
    if collection_user.find_one({"username": create_user_request.username}):
        return {"message": "Username already exists"}

    create_user_model = User(
        firstname=create_user_request.firstname,
        surname=create_user_request.surname,
        email=create_user_request.email,
        username=create_user_request.username,
        password=create_user_request.password,
        user_verification_file_path=create_user_request.verified_file_path,
    )
    create_user(create_user_model)
    subject = "Confirmation of Successful Sign-Up"
    body = f"""Dear {create_user_request.firstname} {create_user_request.surname},
    
We hope this email finds you well.

We are writing to confirm that your sign-up process has been successfully completed. Thank you for choosing to create an account with us.

Your account details are as follows:
Firstname: {create_user_request.firstname}
Surname: {create_user_request.surname}
Email: {create_user_request.email}
Username: {create_user_request.username}

Please note that while your sign-up process has been completed, our team is currently reviewing and verifying accounts. You will receive another email from us once your account has been successfully verified. This process may take some time, but we will keep you updated throughout.

Best regards,
Roof Surface Detection System"""
    try:
        await send_email(create_user_request.email, subject, body)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"There was an error sending email. {e}"
        )

    return {"message": "User created successfully"}


@router.post("/create_admin", status_code=status.HTTP_201_CREATED)
async def create_admin(
    current_user: Annotated[User, Depends(get_current_user)],
    create_admin_request: CreateAdminRequest,
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}
    if collection_user.find_one({"username": create_admin_request.username}):
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Failed to create new account {create_admin_request.username} as Admin due to username exist",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Username already exists"}

    create_admin_model = User(
        firstname=create_admin_request.firstname,
        surname=create_admin_request.surname,
        email=create_admin_request.email,
        username=create_admin_request.username,
        password=create_admin_request.password,
        is_admin=True,
        is_verified=True,
        user_verification_file_path=create_admin_request.verified_file_path,
    )
    create_user(create_admin_model)
    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Create new account {create_admin_request.username} as Admin",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )

    subject = "Congratulations! You've Been Assigned as an Administrator"
    body = f"""Dear {create_admin_request.firstname} {create_admin_request.surname},
    
We are delighted to inform you that you have been assigned as an administrator for Roof Surface Detection System. Congratulations on this new responsibility!

Your login credentials are as follows:

Username: {create_admin_request.username}
Password: {create_admin_request.password}

Please log in using the provided credentials and change your password immediately for security purposes.

As an administrator, you now have access to a range of privileged features and controls within our system. We trust that you will carry out your duties with diligence and professionalism.

Should you encounter any difficulties logging in or have any questions about your new role, please do not hesitate to contact our support team at roofsurfacedetection@gmail.com. We are here to assist you every step of the way.

Thank you for your commitment and contribution to our organization. We look forward to working together to achieve our goals.

Best regards,
Roof Surface Detection System"""
    try:
        await send_email(create_admin_request.email, subject, body)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"There was an error sending email. {e}"
        )

    return {"message": "User created successfully"}


@router.post("/login_token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The identity has not been verified or Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    collection_log.insert_one(
        {
            "actor": form_data.username,
            "message": f"Logged in",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):

    return current_user


# -------------------------------------------------------User-------------------------------------------------------


# GET Request Method for verified user
@router.get("/get_user_verified")
async def get_usr_verified(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    usr_lis = list_serial_user(
        collection_user.find({"is_verified": True, "is_admin": False})
    )
    return usr_lis


# GET Request Method for unverified user
@router.get("/get_user_unverified")
async def get_usr_unverified(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    usr_lis = list_serial_user(
        collection_user.find({"is_verified": False, "is_admin": False})
    )
    return usr_lis


@router.get("/get_admin")
async def get_admin(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    usr_list = list_serial_user(
        collection_user.find({"is_admin": True, "is_verified": True})
    )
    return usr_list


# PUT Request Method for verified user
@router.put("/put_verified")
async def put_user_verified(
    current_user: Annotated[User, Depends(get_current_user)], verified: VerifiedUser
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    who_user = collection_user.find_one({"username": verified.username})

    if who_user:
        collection_user.update_one(
            {"username": verified.username},
            {"$set": {"is_verified": verified.verified}},
        )
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Verify user {verified.username}",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        subject = "Account Verification: Successful Completion"
        body = f"""Dear {who_user['firstname']} {who_user['surname']},
    
We are thrilled to inform you that your account with Roof Surface Detection System has been successfully verified!

This means you now have full access to all the features and benefits our platform offers. You can login and using our detection system.

We sincerely appreciate your patience throughout the verification process. If you have any questions or need further assistance, please don't hesitate to reach out to our support team at roofsurfacedetection@gmail.com.

Thank you for choosing Roof Surface Detection System. We look forward to serving you and hope you have a great experience with us.

Best regards,
Roof Surface Detection System"""
        try:
            await send_email(who_user["email"], subject, body)
        except Exception as e:
            return {"message": f"There was an error sending email. {e}"}

        return {"message": "User Verified"}
    else:
        raise HTTPException(
            status_code=404, detail=f"User '{verified.username}' not found."
        )


# PUT Request Method for change password
@router.put("/put_change_password")
async def put_user_password(
    current_user: Annotated[User, Depends(get_current_user)],
    userchange: UserChangePassword,
):
    who_user = collection_user.find_one({"username": userchange.username})

    if who_user:
        if verify_password(userchange.old_password, who_user["password"]):
            hashed_password = pwd_context.hash(userchange.new_password)
            collection_user.update_one(
                {"username": userchange.username},
                {"$set": {"password": hashed_password}},
            )
            collection_log.insert_one(
                {
                    "actor": current_user.username,
                    "message": f"Change their password",
                    "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
                }
            )
            return {"message": "Password updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Old password is incorrect")
    else:
        raise HTTPException(
            status_code=404, detail=f"User '{userchange.username}' not found."
        )


@router.put("/admin_change_password")
async def put_admin_password(
    current_user: Annotated[User, Depends(get_current_user)],
    adminchange: AdminChangePassword,
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    who_user = collection_user.find_one({"username": adminchange.username})

    if who_user:
        hashed_password = pwd_context.hash(adminchange.new_password)
        collection_user.update_one(
            {"username": adminchange.username}, {"$set": {"password": hashed_password}}
        )
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Change user {adminchange.username} password",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Password Changed"}
    else:
        raise HTTPException(
            status_code=404, detail=f"User '{adminchange.username}' not found."
        )


# Delete User Method
@router.delete("/user")
async def delete_user(
    current_user: Annotated[User, Depends(get_current_user)], user_name: UsernameInput
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}
    await delete_user_permis(UsernameInput(username=str(user_name.username)))
    del_user = collection_user.find_one_and_delete({"username": user_name.username})
    if not del_user["is_verified"]:
        subject = "Account Verification Rejection Notification"
        body = f"""Dear {del_user['firstname']} {del_user['surname']},
    
We regret to inform you that your request for account verification with Roof Surface Detection System has been rejected.

Upon review of the documents you submitted for verification, we found that they did not meet our requirements for verification. We apologize for any inconvenience this may cause.

If you have any questions regarding the rejection or if you wish to re-submit your documents for verification, please don't hesitate to contact our support team at roofsurfacedetection@gmail.com. We'll be happy to assist you further.

Thank you for your understanding.

Best regards,
Roof Surface Detection System"""
        try:
            await send_email(del_user["email"], subject, body)
        except Exception as e:
            return {"message": f"There was an error sending email. {e}"}

    elif del_user["is_verified"] or del_user["is_admin"]:
        subject = "Account Removal Notification"
        body = f"""Dear {del_user['firstname']} {del_user['surname']},
    
We regret to inform you that your account with Roof Surface Detection System has been removed from our system. We understand this may come as a disappointment, and we apologize for any inconvenience this may cause.

For more information regarding the removal of your account and to discuss any potential resolution or clarification, please reach out to our support team at roofsurfacedetection@gmail.com. Our team will be more than happy to assist you further and provide any necessary details.

We appreciate your understanding and cooperation in this matter.

Best regards,
Roof Surface Detection System"""
        try:
            await send_email(del_user["email"], subject, body)
        except Exception as e:
            return {"message": f"There was an error sending email. {e}"}

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Delete user {user_name.username}",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "User Deleted"}


# -------------------------------------------------------Factory-------------------------------------------------------


# GET Request Method for factory information
@router.get("/get_factory_info")
async def get_facto_info(
    current_user: Annotated[User, Depends(get_current_user)], facto_id: str
):
    facto_info = collection_factory.find_one({"_id": ObjectId(facto_id)})
    if facto_info:
        facto_info["_id"] = str(facto_info["_id"])
        return facto_info


# GET Request Method for admin look factory in add factory page
@router.get("/get_admin_factory")
async def get_admin_add_permis(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    facto_lis = list_serial_factory(collection_factory.find({"is_disable": False}))
    return facto_lis


# GET Request Method for admin factory manage page
@router.get("/get_admin_manage_factory")
async def get_admin_manage(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    facto_lis = list_serial_factory(collection_factory.find())
    return facto_lis


# GET Request Method for show factory to user
@router.get("/get_user_factory")
async def get_usr_facto_lis(
    current_user: Annotated[User, Depends(get_current_user)], username: str
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}
    factories_list = []

    who_user = collection_user.find_one({"username": username})
    who_user_id = who_user["_id"]

    which_permis = collection_Permission.find({"user_id": ObjectId(who_user_id)})

    for each_permis in which_permis:

        each_permis_factory_id = str(each_permis["factory_id"])
        find_factory = collection_factory.find_one(
            {"_id": ObjectId(each_permis_factory_id)}
        )
        factory_name = find_factory["factory_name"]
        factory_status = find_factory["is_disable"]
        buildings_lis = []

        if factory_status == False:
            find_building = collection_building.find(
                {"factory_id": each_permis_factory_id}
            )
            for each_building in find_building:
                building_name = each_building["building_name"]
                building_id = str(each_building["_id"])
                buildings_lis.append(
                    {"building_name": building_name, "building_id": building_id}
                )
            factories_list.append(
                {
                    "factory_name": factory_name,
                    "factory_id": each_permis_factory_id,
                    "buildings": buildings_lis,
                }
            )

    return factories_list


# Get Method for admin permission summary
@router.get("/permission_summary")
async def get_permission_summary(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    factory_list = []
    all_factory = collection_factory.find()

    for each_factory in all_factory:
        each_factory_name = str(each_factory["factory_name"])
        each_factory_id = each_factory["_id"]

        user_list = []
        user_count = 0

        find_permission = collection_Permission.find({"factory_id": each_factory_id})
        for each_permission in find_permission:
            user_count += 1
            user_permis_id = each_permission["user_id"]
            find_user = collection_user.find_one({"_id": ObjectId(user_permis_id)})
            user_name = find_user["username"]
            user_list.append({"username": user_name})

        factory_list.append(
            {
                "factory_name": each_factory_name,
                "user_count": user_count,
                "user_permis": user_list,
            }
        )

    return factory_list


# PUT Method for admin change between factory (disable=Ture) and (enable=False)
@router.put("/put_change_facto_status")
async def put_change_facto_status(
    current_user: Annotated[User, Depends(get_current_user)], id_facto: FactoryId
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    find_factory = collection_factory.find_one({"_id": ObjectId(id_facto.facto_id)})

    if find_factory:

        if find_factory["is_disable"] == True:
            collection_factory.update_one(
                {"_id": ObjectId(id_facto.facto_id)}, {"$set": {"is_disable": False}}
            )
            collection_log.insert_one(
                {
                    "actor": current_user.username,
                    "message": f"Activate factory {find_factory['factory_name']}",
                    "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
                }
            )
        elif find_factory["is_disable"] == False:
            collection_factory.update_one(
                {"_id": ObjectId(id_facto.facto_id)}, {"$set": {"is_disable": True}}
            )
            collection_log.insert_one(
                {
                    "actor": current_user.username,
                    "message": f"Disable factory {find_factory['factory_name']}",
                    "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
                }
            )

    return {"message": "Factory Status Updated"}


# POST Request Method
@router.post("/post_factory")
async def post_facto_lis(
    current_user: Annotated[User, Depends(get_current_user)], facto: Factory
):
    collection_factory.insert_one(dict(facto))
    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Create new factory {facto.factory_name}",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )

    return {"message": "Factory Created"}


# Delete Factory and delete every thing about it.
@router.delete("/factory")
async def delete_facto(
    current_user: Annotated[User, Depends(get_current_user)], id_facto: FactoryId
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    find_building_by_facto_id = collection_building.find(
        {"factory_id": str(id_facto.facto_id)}
    )

    for that_building in find_building_by_facto_id:
        which_building_id = that_building["_id"]
        await del_build(BuildingId(build_id=str(which_building_id)))

    await delete_factory_permis(FactoryId(facto_id=str(id_facto.facto_id)))
    rm_factory = collection_factory.find_one_and_delete(
        {"_id": ObjectId(id_facto.facto_id)}
    )
    fac_id = rm_factory["_id"]

    if os.path.exists(f"/app/data/image/{fac_id}") and os.path.isdir(
        f"/app/data/image/{fac_id}"
    ):
        shutil.rmtree(f"/app/data/image/{fac_id}")

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Delete factory {rm_factory['factory_name']}",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "Factory Deleted"}


# -------------------------------------------------------Building--------------------------------------------------------


# GET Request Method for factory information
@router.get("/get_building_info")
async def get_build_info(
    current_user: Annotated[User, Depends(get_current_user)], build_id: str
):
    obj_id = ObjectId(build_id)
    build_info = collection_building.find_one({"_id": obj_id})
    if build_info:
        build_info["_id"] = str(build_info["_id"])
        build_info["factory_id"] = str(build_info["factory_id"])
        return build_info


# GET Request Method for user
@router.get("/get_building")
async def get_build_lis(current_user: Annotated[User, Depends(get_current_user)]):
    build_lis = list_serial_build(collection_building.find())
    return build_lis


@router.put("/change_building_detail")
async def put_building_detail(
    current_user: Annotated[User, Depends(get_current_user)], detail: BuildingDetail
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    which_building = collection_building.find_one({"_id": ObjectId(detail.building_id)})

    if which_building:
        collection_building.update_one(
            {"_id": ObjectId(detail.building_id)},
            {"$set": {"building_length": detail.building_length}},
        )
        collection_building.update_one(
            {"_id": ObjectId(detail.building_id)},
            {"$set": {"building_width": detail.building_width}},
        )
        collection_building.update_one(
            {"_id": ObjectId(detail.building_id)},
            {"$set": {"building_latitude": detail.building_latitude}},
        )
        collection_building.update_one(
            {"_id": ObjectId(detail.building_id)},
            {"$set": {"building_longitude": detail.building_longitude}},
        )
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Update building detail",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Building Detail Changed"}
    else:
        raise HTTPException(status_code=404, detail=f"Building not found.")


# POST Request Method
@router.post("/post_building")
async def post_build_lis(
    current_user: Annotated[User, Depends(get_current_user)],
    build: CreateBuildingRequest,
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    check_exist = collection_building.find_one(
        {"factory_id": build.factory_id, "building_name": build.building_name}
    )

    if check_exist is None:
        building = Building(
            building_name=build.building_name,
            building_length=build.building_length,
            building_width=build.building_width,
            building_latitude=build.building_latitude,
            building_longitude=build.building_longitude,
            data_location="",
            factory_id=build.factory_id,
        )
        build_doc = dict(building)
        collection_building.insert_one(build_doc)
        building = collection_building.find_one(build_doc)
        building_id = str(building["_id"])
        building_path = f"/app/data/image/{build.factory_id}/{building_id}"
        print(building_path)
        collection_building.update_one(
            build_doc, {"$set": {"data_location": building_path}}
        )
        os.makedirs(building_path, exist_ok=True)
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Add new building",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return building_id
    else:
        raise HTTPException(
            status_code=404, detail=f"Building '{build.building_name}' already created."
        )


# Delete Building and all about it
@router.delete("/building")
async def delete_building(
    current_user: Annotated[User, Depends(get_current_user)], id_building: BuildingId
):

    find_history_by_building_id = collection_history.find(
        {"building_id": id_building.build_id}
    )

    for each_history in find_history_by_building_id:
        each_history_id = each_history["_id"]
        await del_his(HistoryId(histo_id=str(each_history_id)))

    rm_building = collection_building.find_one_and_delete(
        {"_id": ObjectId(id_building.build_id)}
    )

    # shutil.rmtree(rm_building['data_location'])

    if os.path.exists(rm_building["data_location"]) and os.path.isdir(
        rm_building["data_location"]
    ):
        shutil.rmtree(rm_building["data_location"])

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Remove building",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "Building Deleted"}


async def del_build(id_building: BuildingId):
    find_history_by_building_id = collection_history.find(
        {"building_id": id_building.build_id}
    )

    for each_history in find_history_by_building_id:
        each_history_id = each_history["_id"]
        await del_his(HistoryId(histo_id=str(each_history_id)))

    rm_building = collection_building.find_one_and_delete(
        {"_id": ObjectId(id_building.build_id)}
    )

    # shutil.rmtree(rm_building['data_location'])

    if os.path.exists(rm_building["data_location"]) and os.path.isdir(
        rm_building["data_location"]
    ):
        shutil.rmtree(rm_building["data_location"])


# -------------------------------------------------------History-----------------------------------------------------


@router.get("/get_history")
async def get_history(
    current_user: Annotated[User, Depends(get_current_user)], id_building: str
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    histo_list = []

    find_history = collection_history.find({"building_id": str(id_building)})

    for each_history in find_history:
        each_history["_id"] = str(each_history["_id"])
        histo_list.append(each_history)

    return histo_list


@router.post("/post_history")
async def post_history(
    current_user: Annotated[User, Depends(get_current_user)], id_building: BuildingId
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    current_datetime_bangkok = getCurTime()
    building = collection_building.find_one({"_id": ObjectId(id_building.build_id)})
    building_dir = building["data_location"]
    history = History(
        create_date=current_datetime_bangkok.strftime("%d-%m-%Y"),
        create_time=current_datetime_bangkok.strftime("%H:%M:%S"),
        building_id=str(id_building.build_id),
        history_path=f'{building_dir}/{current_datetime_bangkok.strftime("%d-%m-%Y_%H-%M-%S")}',
    )
    os.makedirs(
        f'{building_dir}/{current_datetime_bangkok.strftime("%d-%m-%Y_%H-%M-%S")}',
        exist_ok=True,
    )
    history_doc = dict(history)
    collection_history.insert_one(history_doc)
    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Add new history",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "History Created"}


@router.delete("/delete_history")
async def delete_history(
    current_user: Annotated[User, Depends(get_current_user)], id_histo: HistoryId
):

    find_image_by_history_id = collection_Image.find({"history_id": id_histo.histo_id})

    for each_image in find_image_by_history_id:
        each_image_id = each_image["_id"]
        await del_img(ImageId(image_id=str(each_image_id)))

    rm_history = collection_history.find_one_and_delete(
        {"_id": ObjectId(id_histo.histo_id)}
    )

    # shutil.rmtree(rm_history['history_path'])

    if os.path.exists(rm_history["history_path"]) and os.path.isdir(
        rm_history["history_path"]
    ):
        shutil.rmtree(rm_history["history_path"])

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Remove history",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "History Deleted"}


async def del_his(id_histo: HistoryId):

    find_image_by_history_id = collection_Image.find({"history_id": id_histo.histo_id})

    for each_image in find_image_by_history_id:
        each_image_id = each_image["_id"]
        await del_img(ImageId(image_id=str(each_image_id)))

    rm_history = collection_history.find_one_and_delete(
        {"_id": ObjectId(id_histo.histo_id)}
    )

    # shutil.rmtree(rm_history['history_path'])

    if os.path.exists(rm_history["history_path"]) and os.path.isdir(
        rm_history["history_path"]
    ):
        shutil.rmtree(rm_history["history_path"])

    return {"message": "History Deleted"}


# -------------------------------------------------------Image-------------------------------------------------------

@router.get("/get_img/{image_id}")
async def get_image_file(
    # current_user: Annotated[User, Depends(get_current_user)], 
    image_id: str
):
    # if not current_user.is_verified:
    #     collection_log.insert_one(
    #         {
    #             "actor": current_user.username,
    #             "message": f"Trying to access as verified user",
    #             "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
    #         }
    #     )
    #     return {"message": "Not Verified"}

    # Find the image in the database by its ID
    image = collection_Image.find_one({"_id": ObjectId(image_id)})
    if image is None:
        return {"message": "Image not found"}

    # Get the image path from the database
    image_path = image["image_path"]

    # Return the image file as a response
    return FileResponse(image_path)

# GET Request Method for image when need to show image which have defect
@router.get("/get_image")
async def get_image_lis(
    current_user: Annotated[User, Depends(get_current_user)], history_id: str
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    image_list = []

    which_history_id = history_id

    find_image_by_history_id = {"history_id": str(which_history_id)}
    which_image = collection_Image.find(find_image_by_history_id)
    for each_image in which_image:
        which_image_id = each_image["_id"]
        # find_defectlo_by_image_id = {'image_id' : ObjectId(which_image_id)}
        defect_count = collection_DefectLocation.count_documents(
            {
                "image_id": str(which_image_id),
                "is_user_verified": each_image["is_user_verified"],
            }
        )
        which_image_path = each_image["image_path"]
        which_image_x = each_image["x_index"]
        which_image_y = each_image["y_index"]
        image_list.append(
            {
                "image_id": str(which_image_id),
                "image_path": which_image_path,
                "is_verified": each_image["is_user_verified"],
                "defect_count": defect_count,
                "x_offset": which_image_x,
                "y_offset": which_image_y,
            }
        )
    max_x_offset = max(image_list, key=lambda x: x["x_offset"])["x_offset"] + 1
    max_y_offset = max(image_list, key=lambda x: x["y_offset"])["y_offset"] + 1
    offset = {"max_x": max_x_offset, "max_y": max_y_offset}

    return image_list, offset


# POST Request Method
@router.post("/post_image")
async def post_image_lis(
    current_user: Annotated[User, Depends(get_current_user)], img: Image
):
    image_doc = dict(img)
    collection_Image.insert_one(image_doc)

    return {"message": "Image Created"}


# Delete Request Method for image
@router.delete("/image")
async def delete_image_lis(
    current_user: Annotated[User, Depends(get_current_user)], id_image: ImageId
):

    tasks = []
    which_image_id = id_image.image_id
    tasks.append(delete_defectlo_lis(ImageId(image_id=str(which_image_id))))

    await asyncio.gather(*tasks)

    collection_Image.find_one_and_delete({"_id": ObjectId(str(which_image_id))})

    return {"message": "Image Deleted"}


async def del_img(id_image: ImageId):
    tasks = []
    which_image_id = id_image.image_id
    tasks.append(delete_defectlo_lis(ImageId(image_id=str(which_image_id))))

    await asyncio.gather(*tasks)

    collection_Image.find_one_and_delete({"_id": ObjectId(str(which_image_id))})

    return {"message": "Image Deleted"}


# -------------------------------------------------------Defect-------------------------------------------------------


@router.get("/get_defect")
async def get_defect(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    defect_list = []
    find_defect = collection_Defect.find()
    for each_defect in find_defect:
        each_defect["_id"] = str(each_defect["_id"])
        defect_list.append(each_defect)

    return defect_list


@router.post("/post_defect")
async def post_defect(defect: Defect):
    defect_doc = dict(defect)
    collection_Defect.insert_one(defect_doc)


# -------------------------------------------------------DefectLocation-------------------------------------------------------


# GET Request Method for show defect in picture
@router.get("/get_defectLocation")
async def get_defectlo_lis(
    current_user: Annotated[User, Depends(get_current_user)], image_id: str
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    defectlo_lis_ture = []
    defectlo_lis_false = []

    image_id_in_defect = {"image_id": image_id}
    find_defectlo = collection_DefectLocation.find(image_id_in_defect)
    for each_defectlo in find_defectlo:
        if each_defectlo["is_user_verified"] == True:
            each_defectlo["_id"] = str(each_defectlo["_id"])
            defectlo_lis_ture.append(each_defectlo)
        elif each_defectlo["is_user_verified"] == False:
            each_defectlo["_id"] = str(each_defectlo["_id"])
            defectlo_lis_false.append(each_defectlo)

    if len(defectlo_lis_ture) > 0:
        return defectlo_lis_ture
    else:
        return defectlo_lis_false


# GET summary of defect and picture for history summary page
@router.get("/get_summary_user_verified")
async def get_summary_user_verified(
    current_user: Annotated[User, Depends(get_current_user)], histo_id: str
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    summary_list = []
    summary_system = []
    summary_user = []
    defectlo_lis_ture = []
    defectlo_lis_false = []
    count_image = 0
    defect_true_count = 0
    defect_false_count = 0
    any_defect_system = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    any_defect_user = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    find_all_image = collection_Image.find({"history_id": str(histo_id)})

    for each_image in find_all_image:
        count_image += 1
        find_defectlo = collection_DefectLocation.find(
            {"image_id": str(each_image["_id"])}
        )

        for each_defectlo in find_defectlo:
            if each_defectlo["is_user_verified"] == True:
                each_defectlo["_id"] = str(each_defectlo["_id"])
                defectlo_lis_ture.append(each_defectlo)
            elif each_defectlo["is_user_verified"] == False:
                each_defectlo["_id"] = str(each_defectlo["_id"])
                defectlo_lis_false.append(each_defectlo)
                defect_false_count += 1
                any_defect_system[each_defectlo["class_type"]] += 1

        if len(defectlo_lis_ture) > 0:
            defect_true_count += len(defectlo_lis_ture)

            for each_defect in defectlo_lis_ture:
                any_defect_user[each_defect["class_type"]] += 1
        else:
            defect_true_count += len(defectlo_lis_false)

            for each_defect in defectlo_lis_false:
                any_defect_user[each_defect["class_type"]] += 1

        defectlo_lis_ture = []
        defectlo_lis_false = []

    summary_system.append(
        {
            "summary_defect": defect_false_count,
            "birddrop": any_defect_system[0],
            "glue": any_defect_system[1],
            "mud": any_defect_system[2],
            "other": any_defect_system[3],
            "rock": any_defect_system[4],
            "rust": any_defect_system[5],
            "stain": any_defect_system[6],
            "stick": any_defect_system[7],
            "tape": any_defect_system[8],
        }
    )
    summary_user.append(
        {
            "summary_defect": defect_true_count,
            "birddrop": any_defect_user[0],
            "glue": any_defect_user[1],
            "mud": any_defect_user[2],
            "other": any_defect_user[3],
            "rock": any_defect_user[4],
            "rust": any_defect_user[5],
            "stain": any_defect_user[6],
            "stick": any_defect_user[7],
            "tape": any_defect_user[8],
        }
    )
    summary_list.append(
        {
            "photo_count": count_image,
            "summary_systems": summary_system,
            "summary_user": summary_user,
        }
    )

    return summary_list


# POST Request Method for redefine the defect square
@router.post("/post_defectLocation_for_redefine")
async def post_defectlo_lis_redefine(
    current_user: Annotated[User, Depends(get_current_user)],
    defect_with_image: DefectLocationWithImage,
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    Image_post_id = str(defect_with_image.Image_post_id)
    defectlos = defect_with_image.defectlos

    for defectlo in defectlos:
        defectlocation_doc = dict(defectlo)
        defectlocation_doc["image_id"] = Image_post_id
        defectlocation_doc["is_user_verified"] = True
        class_type = defectlocation_doc["class_type"]
        class_data = collection_Defect.find({"defect_class": class_type})
        for each_doc in class_data:
            class_name = each_doc["defect_class_name"]
        defectlocation_doc["class_name"] = class_name
        collection_DefectLocation.insert_one(defectlocation_doc)
        defectlocation_doc.clear()

    collection_Image.update_one(
        {"_id": ObjectId(Image_post_id)}, {"$set": {"is_user_verified": True}}
    )
    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Update and verify defect",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "DefectLocation Verified"}


# POST Request Method for use model detection
# Use image path for parameter
# @router.post("/post_defectLocation_for_model")


# Delete Request Method for renew defect
@router.delete("/delete_for_renew")
async def delete_for_renew(
    current_user: Annotated[User, Depends(get_current_user)], id_image: ImageId
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    find_defect = collection_DefectLocation.find({"image_id": id_image.image_id})

    for each_defect in find_defect:
        if each_defect["is_user_verified"] == True:
            collection_DefectLocation.find_one_and_delete(
                {"_id": ObjectId(each_defect["_id"])}
            )

    return {"message": "DefectLocation Verify Renewed"}


# Delete Request Method for redefind defect by image_id
@router.delete("/defectlo")
async def delete_defectlo_lis(id_image: ImageId):

    defectloc_image = collection_DefectLocation.find(
        {"image_id": str(id_image.image_id)}
    )

    for each_doc in defectloc_image:
        defectlo_id = each_doc["_id"]
        collection_DefectLocation.find_one_and_delete({"_id": ObjectId(defectlo_id)})

    return {"message": "DefectLocation Deleted"}


# -------------------------------------------------------Permission-------------------------------------------------------
# GET Request Method
@router.get("/get_permission_factory")
async def get_permis_factory(
    current_user: Annotated[User, Depends(get_current_user)], facto_id: str
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    obj_id = ObjectId(facto_id)
    find_permission = collection_Permission.find({"factory_id": obj_id})

    user_list = []
    for each_permission in find_permission:
        that_user_id = str(each_permission["user_id"])
        which_user = {"_id": ObjectId(that_user_id)}
        find_user = collection_user.find_one(which_user)
        print(find_user)
        if find_user:
            find_user["_id"] = str(find_user["_id"])
            user_list.append(find_user)

    return user_list


# Get Method for show verified user who dont have permission in that factory
@router.get("/get_not_permission_factory")
async def get_no_permis_facto(
    current_user: Annotated[User, Depends(get_current_user)], facto_id: str
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    obj_id = ObjectId(facto_id)
    verified_users = collection_user.find({"is_verified": True, "is_admin": False})

    user_list = []
    for each_verified_users in verified_users:
        each_verified_users["_id"] = str(each_verified_users["_id"])
        user_list.append(each_verified_users)

    find_permission = collection_Permission.find({"factory_id": obj_id})
    for each_permission in find_permission:
        that_user_id = str(each_permission["user_id"])
        that_user_id = str(each_permission["user_id"])
        for user in user_list:
            if user["_id"] == that_user_id:
                user_list.remove(user)

    return user_list


# Post Request Method
@router.post("/post_permission")
async def post_permis_lis(
    current_user: Annotated[User, Depends(get_current_user)], permis: Permission
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    permis_doc = dict(permis)
    user_post_name = permis_doc["username"]
    factory_post_name = permis_doc["factory_name"]
    factory_post_details = permis_doc["factory_details"]

    find_user = collection_user.find_one({"username": user_post_name})
    if find_user:
        if not find_user["is_verified"]:
            return {"message": "User not verified"}
    that_user_id = find_user["_id"]

    which_factory = {
        "$and": [
            {"factory_name": factory_post_name},
            {"factory_details": factory_post_details},
        ]
    }
    find_factory = collection_factory.find_one(which_factory)
    that_factory_id = find_factory["_id"]

    permis_doc["user_id"] = that_user_id  # user_id
    permis_doc["factory_id"] = that_factory_id  # factory_id
    permis_doc.pop("username", None)
    permis_doc.pop("factory_name", None)
    permis_doc.pop("factory_details", None)
    collection_Permission.insert_one(permis_doc)

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Add user {user_post_name} to factory {factory_post_name}",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "Permission Created"}


@router.delete("/del_permission")
async def delete_permission(
    current_user: Annotated[User, Depends(get_current_user)], user_fac: UserFac
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    permis = collection_Permission.find_one_and_delete(
        {
            "user_id": ObjectId(str(user_fac.user_id)),
            "factory_id": ObjectId(str(user_fac.fac_id)),
        }
    )
    if not permis:
        return {"message", f"Permission {user_fac} not found"}

    user = collection_user.find_one({"_id": ObjectId(user_fac.user_id)})
    fac = collection_factory.find_one({"_id": ObjectId(user_fac.fac_id)})

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Remove user {user['username']} from factory {fac['factory_name']}",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )
    return {"message": "Permission Deleted"}


# Delete Permission Method by user
@router.delete("/permis_user")
async def delete_user_permis(permis_username: UsernameInput):
    print("delete_user_permis" + " : " + permis_username.username)
    who_user_username = {"username": str(permis_username.username)}
    who_user = collection_user.find(who_user_username)

    for that_user in who_user:
        who_user_id = that_user["_id"]
        which_permis = collection_Permission.find({"user_id": ObjectId(who_user_id)})
        for each_permis in which_permis:
            collection_Permission.find_one_and_delete(each_permis)

    return {"message": "Permission Deleted"}


@router.delete("/permis_facto")
async def delete_factory_permis(id_facto: FactoryId):

    which_permis = collection_Permission.find(
        {"factory_id": ObjectId(id_facto.facto_id)}
    )
    for each_permis in which_permis:
        collection_Permission.find_one_and_delete(each_permis)

    return {"message": "Permission Deleted"}


# -------------------------------------------------------File-------------------------------------------------------
# GET Request Method
@router.post("/get_verification_file")
async def get_verification_file(
    current_user: Annotated[User, Depends(get_current_user)], username: UsernameInput
):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as admin",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    user = collection_user.find_one({"username": username.username})
    user_path = user["user_verification_file_path"]
    filename = user_path.split("/")[-1].split(".")[-1]

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Download user {username.username} verification file",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )

    return FileResponse(
        path=user_path,
        media_type="application/octet-stream",
        filename=f"{username.username}.{filename}",
    )


# -------------------------------------------------------Video-------------------------------------------------------
def extract_metadata_from_srt(srt_file):
    metadata = []
    prev_lat = None
    prev_lon = None
    for srt in srt_file:
        with open(f"{srt}.SRT", "r") as f:
            lines = f.readlines()
            metadata_block = {}
            for line in lines:
                if line.strip():  # If line is not empty
                    if re.match(r"^\d+$", line, re.MULTILINE):
                        metadata_block["frame"] = line.replace("\n", "")
                    if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3}", line):
                        metadata_block["datetime"] = line.replace("\n", "")
                    matches = re.findall(r"\[(.*?)\]", line)
                    for match in matches:
                        if "," in match:
                            value = match.split(",", 1)
                            key1, value1 = value[0].split(":", 1)
                            key2, value2 = value[1].split(":", 1)
                            metadata_block[key1.strip()] = value1.strip()
                            metadata_block[key2.strip()] = value2.strip()
                        elif "alt" in match:
                            key1, value1, key2, value2 = match.split()
                            key1 = key1.replace(":", "")
                            key2 = key2.replace(":", "")
                            metadata_block[key1.strip()] = value1.strip()
                            metadata_block[key2.strip()] = value2.strip()
                        else:
                            key, value = match.split(":", 1)
                            metadata_block[key.strip()] = value.strip()
                else:
                    if "latitude" in metadata_block and "longitude" in metadata_block:
                        if (
                            not prev_lat
                            or abs(prev_lat - float(metadata_block["latitude"]))
                            >= 0.000005
                            or abs(prev_lon - float(metadata_block["longitude"]))
                            >= 0.000005
                        ):
                            metadata.append(metadata_block)
                            prev_lat = float(metadata_block["latitude"])
                            prev_lon = float(metadata_block["longitude"])
                        metadata_block = {}
            # Append the last metadata block
            if "latitude" in metadata_block and "longitude" in metadata_block:
                if (
                    abs(prev_lat - float(metadata_block["latitude"])) >= 0.000005
                    or abs(prev_lon - float(metadata_block["longitude"])) >= 0.000005
                ):
                    metadata.append(metadata_block)
                    prev_lat = float(metadata_block["latitude"])
                    prev_lon = float(metadata_block["longitude"])
    return metadata


def calculate_corners(
    frame, center_latitude, center_longitude, width=2.68, height=3.531
):
    # Earth's radius in meters
    earth_radius = 6378137.0

    # 1 meter in degrees (approximately)
    meter_degrees = 1.0 / (2 * math.pi * earth_radius / 360)

    # Convert width and height from meters to degrees
    width_degrees = width * meter_degrees
    height_degrees = height * meter_degrees

    # Calculate corner coordinates
    top_left_latitude = center_latitude + (height_degrees / 2)
    top_left_longitude = center_longitude - (width_degrees / 2)
    bottom_right_latitude = center_latitude - (height_degrees / 2)
    bottom_right_longitude = center_longitude + (width_degrees / 2)

    return {
        "frame": frame,
        "center": (center_latitude, center_longitude),
        "top_left": (top_left_latitude, top_left_longitude),
        "bottom_right": (bottom_right_latitude, bottom_right_longitude),
        "top_right": (top_left_latitude, bottom_right_longitude),
        "bottom_left": (bottom_right_latitude, top_left_longitude),
    }


def rectangles_overlap(rect1, rect2, ratio):
    # Extract coordinates of rectangles
    A_maxY, A_minX = rect1["top_left"]
    A_minY, A_maxX = rect1["bottom_right"]

    B_maxY, B_minX = rect2["top_left"]
    B_minY, B_maxX = rect2["bottom_right"]

    # Calculate the width and height of each rectangle
    width_A = A_maxX - A_minX
    height_A = A_maxY - A_minY
    width_B = B_maxX - B_minX
    height_B = B_maxY - B_minY

    # Calculate the coordinates of the intersection rectangle
    inter_width = min(A_maxX, B_maxX) - max(A_minX, B_minX)
    inter_height = min(A_maxY, B_maxY) - max(A_minY, B_minY)

    # If there is no intersection, return False
    if inter_width <= 0 or inter_height <= 0:
        return False

    # return True
    # Calculate the area of intersection
    inter_area = inter_width * inter_height

    # # Calculate the total area of each rectangle
    area_A = width_A * height_A
    area_B = width_B * height_B

    # # Calculate the minimum allowed overlap area (10% of the smaller rectangle)
    min_overlap_area = min(area_A, area_B) * ratio

    # # Check if the intersection area is greater than or equal to the minimum allowed overlap area
    if inter_area >= min_overlap_area:
        return True
    else:
        return False


def find_non_overlapping_rectangles(rectangles):
    non_overlapping_rectangles = []
    num_rectangles = len(rectangles)
    moving_dir = None
    check_dir = tuple(
        abs(t1 - t2) for t1, t2 in zip(rectangles[0]["center"], rectangles[1]["center"])
    )
    if check_dir[0] >= 0.000005:
        moving_dir = "y"
    elif check_dir[1] >= 0.000005:
        moving_dir = "x"
    cur_dir = None
    change_dir = False
    row = 1
    output = {}
    output[str(row)] = []
    for i in range(1, num_rectangles):
        if (
            tuple(
                abs(t1 - t2)
                for t1, t2 in zip(rectangles[i - 1]["center"], rectangles[i]["center"])
            )[0]
            >= 0.000005
        ):
            cur_dir = "y"
        elif (
            tuple(
                abs(t1 - t2)
                for t1, t2 in zip(rectangles[i - 1]["center"], rectangles[i]["center"])
            )[1]
            >= 0.000005
        ):
            cur_dir = "x"

        if moving_dir == cur_dir and not change_dir:
            is_overlap = False
            for j in non_overlapping_rectangles:
                if rectangles_overlap(rectangles[i], j, 0):
                    is_overlap = True
                    break
            if not is_overlap and rectangles[i] not in non_overlapping_rectangles:
                non_overlapping_rectangles.append(rectangles[i])
                output[str(row)].append(rectangles[i])
                non_overlapping_rectangles = sorted(
                    non_overlapping_rectangles, key=lambda x: int(x["frame"])
                )
        else:
            is_overlap = False
            for j in non_overlapping_rectangles:
                if rectangles_overlap(rectangles[i - 1], j, 0.5):
                    is_overlap = True
                    break
            if rectangles[i - 1] not in non_overlapping_rectangles and not change_dir:
                if is_overlap:
                    non_overlapping_rectangles = non_overlapping_rectangles[:-1]
                    output[str(row)] = output[str(row)][:-1]
                if (
                    abs(
                        int(non_overlapping_rectangles[-1]["frame"])
                        - int(rectangles[i - 1]["frame"])
                    )
                    < 700
                ):
                    non_overlapping_rectangles.append(rectangles[i - 1])
                    output[str(row)].append(rectangles[i - 1])
                elif (
                    not is_overlap
                    and abs(
                        int(non_overlapping_rectangles[-1]["frame"])
                        - int(rectangles[i - 1]["frame"])
                    )
                    >= 700
                ):
                    non_overlapping_rectangles = non_overlapping_rectangles[:-1]
                non_overlapping_rectangles = sorted(
                    non_overlapping_rectangles, key=lambda x: int(x["frame"])
                )
            change_dir = True
            if (
                moving_dir == cur_dir
                and rectangles[i - 1] not in non_overlapping_rectangles
                and change_dir
            ):
                if is_overlap:
                    non_overlapping_rectangles = non_overlapping_rectangles[:-1]
                    output[str(row)] = output[str(row)][:-1]
                non_overlapping_rectangles.append(rectangles[i - 1])
                row += 1
                output[str(row)] = []
                output[str(row)].append(rectangles[i - 1])
                non_overlapping_rectangles = sorted(
                    non_overlapping_rectangles, key=lambda x: int(x["frame"])
                )
                change_dir = False

    if row % 2 == 0:
        first_rect = output[str(row)][0]
        output[str(row)] = output[str(row)][:1]
        for i in range(rectangles.index(first_rect), num_rectangles):
            is_overlap = False
            for j in output[str(row)]:
                if rectangles_overlap(rectangles[i], j, 0.01):
                    is_overlap = True
                    break
            if not is_overlap and rectangles[i] not in output[str(row)]:
                output[str(row)].append(rectangles[i])
    max_length = max(len(value) for value in output.values())
    output = {key: value for key, value in output.items() if len(value) == max_length}

    return output


def add_gps_metadata(image_file, metadata):
    photo = gpsphoto.GPSPhoto(image_file)
    info = gpsphoto.GPSInfo(
        (float(metadata["latitude"]), float(metadata["longitude"])),
        alt=int(float(metadata["abs_alt"])),
        timeStamp=datetime.strptime(metadata["datetime"], "%Y-%m-%d %H:%M:%S.%f"),
    )
    photo.modGPSData(info, image_file)


def process_video(video_file, data, count, out_dir, index):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    history = collection_history.find_one({"history_path": out_dir})
    history_id = str(history["_id"])
    frame_count = count
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((data[frame_count])["frame"]))
        ret, frame = cap.read()
        if not ret:
            break
        # Save the frame as an image
        img_out = f"{out_dir}/frame_{frame_count}.jpg"
        cv2.imwrite(img_out, frame)

        add_gps_metadata(img_out, data[frame_count])

        image = Image(
            image_path=img_out,
            x_index=index[frame_count][0],
            y_index=index[frame_count][1],
            history_id=history_id,
        )
        image_doc = dict(image)
        collection_Image.insert_one(image_doc)

        frame_count += 1
        if frame_count >= len(data):
            break

    cap.release()
    return frame_count


@router.post("/upload_video_srt_file")
async def upload_video_srt(
    current_user: Annotated[User, Depends(get_current_user)], fileList: List[UploadFile]
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    file_path = str(uuid.uuid4())
    os.makedirs(f"/app/data/video/{file_path}", exist_ok=True)
    for file in fileList:
        file_dir = f"/app/data/video/{file_path}/{file.filename}"
        try:
            contents = await file.read()
            with open(file_dir, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(
                status_code=404, detail=f"There was an error uploading the file. {e}"
            )
        finally:
            await file.close()

    return {"path": f"/app/data/video/{file_path}"}


@router.post("/extract_video")
async def extract_video(
    current_user: Annotated[User, Depends(get_current_user)], path: ExtractVideo
):
    if not current_user.is_verified:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Verified"}

    frame_count = 0
    directory = path.input_dir
    file_names_without_extension = set()

    for file in os.listdir(directory):
        file_name, file_extension = os.path.splitext(file)
        if file_extension != "":
            file_names_without_extension.add(f"{directory}/{file_name}")

    for file in file_names_without_extension:
        metadata = extract_metadata_from_srt(file_names_without_extension)
        rect = []
        for coor in metadata:
            corner = calculate_corners(
                int(coor["frame"]), float(coor["latitude"]), float(coor["longitude"])
            )
            rect.append(corner)
        result = [list(i) for i in find_non_overlapping_rectangles(rect).values()]
        row = 0
        ascending_lat = (
            tuple(
                t1 - t2
                for t1, t2 in zip(result[0][0]["center"], result[0][-1]["center"])
            )[0]
            < 0
        )
        ascending_lon = (
            tuple(
                t1 - t2
                for t1, t2 in zip(result[0][0]["center"], result[0][-1]["center"])
            )[1]
            < 0
        )
        data = [[], []]
        for i in result:
            if (
                tuple(abs(t1 - t2) for t1, t2 in zip(i[0]["center"], i[-1]["center"]))[
                    0
                ]
                >= 0.00005
            ):
                if ascending_lat:
                    i = sorted(i, key=lambda x: float(x["center"][0]))
                else:
                    i = sorted(i, key=lambda x: float(x["center"][0]), reverse=True)
                for j in range(0, len(i)):
                    data[0].append(i[(len(i) - 1 - j)])
                    data[1].append((row, j))
            elif (
                tuple(abs(t1 - t2) for t1, t2 in zip(i[0]["center"], i[-1]["center"]))[
                    1
                ]
                >= 0.00005
            ):
                if ascending_lon:
                    i = sorted(i, key=lambda x: float(x["center"][1]))
                else:
                    i = sorted(i, key=lambda x: float(x["center"][1]), reverse=True)
                for j in range(len(i) - 1, -1, -1):
                    data[0].append(i[(len(i) - 1 - j)])
                    data[1].append((row, j))
            row += 1
        final_data = []
        for index in data[0]:
            for d in metadata:
                if int(d["frame"]) == int(index["frame"]):
                    final_data.append(d)
        frame_count = process_video(
            f"{directory}/{file_name}.MP4",
            final_data,
            frame_count,
            path.output_dir,
            data[1],
        )

    # shutil.rmtree(path.input_dir)

    if os.path.exists(path.input_dir) and os.path.isdir(path.input_dir):
        shutil.rmtree(path.input_dir)

    await post_defectlo_lis_model(path.output_dir)

    collection_log.insert_one(
        {
            "actor": current_user.username,
            "message": f"Upload and process video",
            "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
        }
    )

    return {"message": "Extract and Process Video Completed"}


async def post_defectlo_lis_model(path: str):
    # get position's image path and model path
    img_path = glob.glob(f"{path}/*.jpg")
    # print(img_path)
    # img_path = os.path.join(path.history_path) #image path for model
    model_path = os.path.join("bestv40.pt")
    model = YOLO(model_path)
    for image in img_path:
        img = cv2.imread(image)
        print(image)
        # print(image.split('\\'))
        # split_str = image.split("\\")
        find_image = collection_Image.find_one({"image_path": f"{image}"})

        results = model.predict(img, conf=0.3)

        for r in results:
            annotator = Annotator(img, font_size=0.1)

            boxes = r.boxes
            for box in boxes:
                f = box.xywhn[0]
                Xn = float(f[0])
                Yn = float(f[1])
                Weighthn = float(f[2])
                Heighthn = float(f[3])

                g = box.cls[0]
                clazz = int(g)

                defect_location = DefectLocation(
                    class_type=clazz,
                    x=Xn,
                    y=Yn,
                    w=Weighthn,
                    h=Heighthn,
                    is_user_verified=False,
                )

                # Convert DefectLocation instance to dictionary
                defectlocation_doc = defect_location.dict()

                defectlocation_doc["image_id"] = str(find_image["_id"])  # image_id
                class_type = defectlocation_doc["class_type"]
                class_data = collection_Defect.find({"defect_class": class_type})
                for each_doc in class_data:
                    class_name = each_doc["defect_class_name"]
                defectlocation_doc["class_name"] = class_name
                collection_DefectLocation.insert_one(defectlocation_doc)
                defectlocation_doc.clear()
        img = annotator.result()

    collection_history.find_one_and_update(
        {"history_path": path}, {"$set": {"is_process": True}}
    )


# -------------------------------------------------------Log-------------------------------------------------------
@router.get("/log")
async def log(current_user: Annotated[User, Depends(get_current_user)]):
    if not current_user.is_admin:
        collection_log.insert_one(
            {
                "actor": current_user.username,
                "message": f"Trying to access as verified user",
                "timestamp": getCurTime().strftime("%d-%m-%Y_%H-%M-%S"),
            }
        )
        return {"message": "Not Admin"}

    log = collection_log.find()
    list_log = list_serial_log(log)
    list_log.reverse()

    return list_log


# -------------------------------------------------------Email-------------------------------------------------------
async def send_email(email_receiver, subject, body):
    email_sender = "roofsurfacedetection@gmail.com"
    email_password = "zivu baqk gyun lprm"

    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
        smtp.quit()
