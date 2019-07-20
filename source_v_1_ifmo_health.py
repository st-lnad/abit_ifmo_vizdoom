from vizdoom import *
import numpy as np
import cv2
import random
import time
import os

# Game settings
game = DoomGame()
game.load_config("/home/leo/0/VizDoom_v_1_0/scenarios/" + "health_gathering_supreme" + ".cfg")
game.set_doom_game_path("/home/leo/0/VizDoom_v_1_0/sprites/" + "freedoom2" + ".wad")
game.set_screen_format(ScreenFormat.BGR24)
game.set_screen_resolution(ScreenResolution.RES_320X240)  # 31-32 pix HUD


# Actions settings
ATTACK = 0
USE = 1
JUMP = 2
CROUCH = 3
TURN180 = 4
ALTATTACK = 5
RELOAD = 6
ZOOM = 7

SPEED = 8
STRAFE = 9

MOVE_RIGHT = 10
MOVE_LEFT = 11
MOVE_BACKWARD = 12
MOVE_FORWARD = 13
TURN_RIGHT = 14
TURN_LEFT = 15
LOOK_UP = 16
LOOK_DOWN = 17
MOVE_UP = 18
MOVE_DOWN = 19
LAND = 20

SELECT_WEAPON1 = 21
SELECT_WEAPON2 = 22
SELECT_WEAPON3 = 23
SELECT_WEAPON4 = 24
SELECT_WEAPON5 = 25
SELECT_WEAPON6 = 26
SELECT_WEAPON7 = 27
SELECT_WEAPON8 = 28
SELECT_WEAPON9 = 29
SELECT_WEAPON0 = 30

SELECT_NEXT_WEAPON = 31
SELECT_PREV_WEAPON = 32
DROP_SELECTED_WEAPON = 33

ACTIVATE_SELECTED_ITEM = 34
SELECT_NEXT_ITEM = 35
SELECT_PREV_ITEM = 36
DROP_SELECTED_ITEM = 37

LOOK_UP_DOWN_DELTA = 38
TURN_LEFT_RIGHT_DELTA = 39
MOVE_FORWARD_BACKWARD_DELTA = 40
MOVE_LEFT_RIGHT_DELTA = 41
MOVE_UP_DOWN_DELTA = 42
# [action[ATTACK][i]+action[USE][i] for i in range(ButtonsCount)]

ButtonsCount = 43
action = [0] * ButtonsCount
n = []
game.set_available_buttons(n)
for i in range(0, ButtonsCount):
    game.add_available_button(Button(i))
    action[i] = [0] * ButtonsCount
    action[i][i] = 1

game.init()

# Visor settings
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
Enemy_detector = cv2.ORB_create(15)
orbM = cv2.ORB_create(50)
Enemy_src_raw = cv2.imread("/home/leo/0/VizDoom_v_1_0/freedoom-png-master/sprites/media0.png")
Enemy_src = cv2.cvtColor(Enemy_src_raw, cv2.COLOR_BGRA2RGB)
Enemy_src = cv2.convertScaleAbs(Enemy_src, cv2.CV_8UC3)
Enemy_src = cv2.resize(Enemy_src, (80, 102))

Filter_src_raw = cv2.imread("/home/leo/0/VizDoom_v_1_0/freedoom-png-master/flats/mflr8_1.png")
Filter_src = cv2.cvtColor(Filter_src_raw, cv2.COLOR_BGRA2RGB)
Filter_src = cv2.convertScaleAbs(Filter_src, cv2.CV_8UC3)
Filter_src = cv2.resize(Filter_src, (80, 102))

Filter2_src_raw = cv2.imread("/home/leo/0/VizDoom_v_1_0/freedoom-png-master/flats/nukage2.png")
Filter2_src = cv2.cvtColor(Filter2_src_raw, cv2.COLOR_BGRA2RGB)
Filter2_src = cv2.convertScaleAbs(Filter2_src, cv2.CV_8UC3)
Filter2_src = cv2.resize(Filter2_src, (80, 102))

Enemy_keypoints1, Enemy_descriptors1 = Enemy_detector.detectAndCompute(Enemy_src, None)
Enemy_descriptors1 = cv2.convertScaleAbs(Enemy_descriptors1, cv2.CV_32F)


Enemy_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)

# Heal
"""
Heal_src = cv2.imread("/home/leo/0/VizDoom_v_1_0/freedoom-png-master/sprites/media0.png", cv2.IMREAD_GRAYSCALE)
Heal_detector = cv2.ORB_create(100)
Heal_keypoints1, Heal_descriptors1 = Heal_detector.detectAndCompute(Heal_src, None)
Heal_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)"""

# Debug settings

my_actions = [action[TURN_LEFT], action[MOVE_FORWARD]]

# game
episodes = 100
best = 0
worth = 100
all_time = 0
n = 0
for i in range(episodes):

    game.new_episode()
    n +=1
    start = time.time()
    while not game.is_episode_finished():
        state = game.get_state()

        img = state.screen_buffer
        # cv2.imwrite("1111.png", img)
        hud = 76  # определять на глаз
        frame_top = img_hud = img.copy()
        frame_top = frame_top[100:240, 0:320]
        #frame_top =  cv2.blur(frame_top, (5,5))
        '''frame_top = frame_top[0:250, 0:640]
            frame_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2GRAY)
            # #cv2.imshow("Frame", frame)
            img_hud = img_hud[480 - hud:480, 0:640]

            #cv2.imshow("HUD", img_hud)'''

        # Enemy
        d = cv2.matchTemplate(frame_top, Enemy_src, cv2.TM_CCOEFF)
        cv2.normalize(d, d, 0, 1, cv2.NORM_MINMAX, -1)
        fd = cv2.matchTemplate(frame_top, Filter_src, cv2.TM_CCOEFF)
        cv2.normalize(fd, fd, 0, 1, cv2.NORM_MINMAX, -1)
        fd2 = cv2.matchTemplate(frame_top, Filter2_src, cv2.TM_CCOEFF)
        cv2.normalize(fd2, fd2, 0, 1, cv2.NORM_MINMAX, -1)
        d -= fd / 3
        d -= fd2 / 3
        #cv2.imshow("d",d)
        Enemy_keypoints2, Enemy_descriptors2 = orbM.detectAndCompute(frame_top, None)
        Enemy_descriptors2 = cv2.convertScaleAbs(Enemy_descriptors2, cv2.CV_32F)
        #cv2.drawKeypoints(frame_top, Enemy_keypoints2, frame_top, 1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if len(Enemy_keypoints1) > 0 and len(Enemy_keypoints2) > 15:
            matches = []
            matches = matcher.knnMatch(Enemy_descriptors2, Enemy_descriptors1, 5)

            """for v in matches:
                if len(v)>2:
                    cv2.rectangle(frame_top,
                                  Enemy_keypoints2[v[0].queryIdx].pt,
                                  Enemy_keypoints2[v[len(v) - 1].queryIdx].pt,
                                  1, 4, 9)"""
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(d)

        #print(cv2.minMaxLoc(d))
        x, y = maxLoc

        if len(Enemy_keypoints2) < 25:
            game.make_action(action[TURN180])
            continue

        if y<90:
            if x > 180+25:
                game.make_action([action[MOVE_FORWARD][i]+action[TURN_RIGHT][i] for i in range(ButtonsCount)])
            elif x > 180-25:
                game.make_action([action[MOVE_FORWARD][i]+action[TURN_LEFT][i] for i in range(ButtonsCount)])
            else:
                game.make_action(action[MOVE_FORWARD])
        else:
            if x > 180+15:
                game.make_action([action[MOVE_BACKWARD][i]+action[TURN_RIGHT][i] for i in range(ButtonsCount)])
            elif x > 180-15:
                game.make_action([action[MOVE_BACKWARD][i]+action[TURN_LEFT][i] for i in range(ButtonsCount)])
            else:
                game.make_action(action[MOVE_FORWARD])

        if Enemy_descriptors2 is None:

            continue

        Enemy_matches = Enemy_matcher.match(Enemy_descriptors1, Enemy_descriptors2)
        img_matches = np.empty(
            (max(Enemy_src.shape[0], frame_top.shape[0]), Enemy_src.shape[1] + frame_top.shape[1], 3),
            dtype=np.uint8)
        # cv2.drawMatches(Enemy_src, Enemy_keypoints1, frame_top, Enemy_keypoints2, Enemy_matches, img_matches)

        needle = Enemy_src
        haystack = frame_top

        frame_top = cv2.drawKeypoints(frame_top, Enemy_keypoints2, frame_top)
        game.make_action(random.choice(my_actions))

        # Heal
        """img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[240:480, 0:640]
            Heal_keypoints2, Heal_descriptors2 = Heal_detector.detectAndCompute(img, None)
            Heal_matches = Heal_matcher.match(Heal_descriptors1, Heal_descriptors2)

            img_matches1 = np.empty(
                (max(Heal_src.shape[0], img.shape[0]), Heal_src.shape[1] + img.shape[1], 3),
                dtype=np.uint8)
            cv2.drawMatches(Heal_src, Heal_keypoints1, img, Heal_keypoints2, Heal_matches, img_matches1)"""

        #cv2.imshow('Matches', frame_top)
        #cv2.imshow('Matches_heal', img_matches1)


        cv2.waitKey(5)
    print("State #" + str(n))
    end =  time.time()-start
    all_time += end
    if best<end:
        best = end
    if worth>end:
        worth = end
    print("Reward:", round(end, 2), "sec")
    print("Best:", round(best, 2), "sec")
    print("Worst:", round(worth, 2), "sec")
    print("Middle:", round(all_time/n, 2), "sec")
    print("=====================")
cv2.destroyAllWindows()
