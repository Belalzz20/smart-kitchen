/*
    Smart Kitchen - TUIO + Bluetooth Personalized Cooking Assistant

    Features:
    - Scan ingredient or physical kitchen object once and keep it saved
    - Use marker rotation to estimate quantity
    - Show calories, health score, meal suggestions, missing items, and cooking steps
    - Support physical kitchen objects: Spoon, Pot, Knife
    - Circular menu with marker 20
    - Login by marker 25 OR Bluetooth detected profile
    - Load user profile from SmartKitchenProfiles.txt
    - Read active Bluetooth user from active_bluetooth_device.txt
    - Hide blocked markers and personalize meal suggestions

    Context CRUD uses tangible TUIO markers, not keyboard shortcuts:
    Gaze Tracking supports reports/hits/analysis and adaptive interface via Python socket 5001.
    Marker 11 = Create/Add context recipe
    Marker 12 = Update nearest context recipe
    Marker 13 = Delete nearest context recipe
    Marker 14 = Reload SmartKitchenContent.txt
*/

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Diagnostics;
using System.Windows.Forms;
using TUIO;

public class SmartKitchenDemo : Form, TuioListener
{
    public static int width;
    public static int height;

    // Facial emotion adaptive interface
    private string currentEmotion = "Neutral";
    private Color adaptiveBackground = Color.FromArgb(250, 248, 243);
    private bool emotionHappyMode = false;
    private bool emotionSadMode = false;
    private bool emotionAngryMode = false;

    private TuioClient client;
    private Dictionary<long, TuioDemoObject> objectList = new Dictionary<long, TuioDemoObject>();
    private readonly object objectSync = new object();

    private Dictionary<int, TuioDemoObject> scannedMarkers = new Dictionary<int, TuioDemoObject>();
    private readonly object scannedSync = new object();
    private List<int> scanHistory = new List<int>();

    // YOLO virtual rotation: Python sends OBJECT_ROT:<name>;ANGLE=<radians> so YOLO items can change quantity like TUIO marker rotation.
    private Dictionary<int, float> yoloMarkerAngles = new Dictionary<int, float>();

    private bool showHealthTips = true;
    private bool showMealSteps = true;
    private bool isLoggedIn = false;

    private const int LOGIN_MARKER_ID = 25;
    private const int SIGNUP_NAME_MARKER_ID = 60;
    private const int SIGNUP_CHEF_MARKER_ID = 61;
    private const int SIGNUP_CLIENT_MARKER_ID = 62;
    private const int SIGNUP_CAPTURE_MARKER_ID = 63;
    private const int SIGNUP_COMMAND_PORT = 5056;
    private const int MENU_MARKER_ID = 20;
    private const int MENU_HOLD_MS = 1000;

    // Context CRUD control markers. These are tangible actions, so the user does not need the keyboard.
    private const int CRUD_ADD_MARKER_ID = 11;
    private const int CRUD_UPDATE_MARKER_ID = 12;
    private const int CRUD_DELETE_MARKER_ID = 13;
    private const int CRUD_RELOAD_MARKER_ID = 14;
    private const int CRUD_ACTION_COOLDOWN_MS = 1200;

    private const string PROFILE_FILE = "SmartKitchenProfiles.txt";
    private const string ACTIVE_BT_FILE = "active_bluetooth_device.txt";
    private const string FACE_LOGIN_FILE = "active_face_user.txt";
    private const string SOCKET_HOST = "127.0.0.1";
    private const int SOCKET_PORT = 5055;
    private const string GESTURE_SOCKET_HOST = "127.0.0.1";
    private const int GESTURE_SOCKET_PORT = 5000;
    private const int BLUETOOTH_PRESENCE_TIMEOUT_SECONDS = 25;

    private TcpClient gestureClient;
    private StreamReader gestureReader;
    private Thread gestureThread;
    private volatile bool gestureRunning = false;
    private string lastGestureCommand = "Waiting for Python gesture connection...";
    private int currentRecipeStepIndex = 0;
    private string currentRecipeName = "";

    private string profileFilePath = "";
    private string activeBluetoothFilePath = "";
    private string activeFaceFilePath = "";
    private string contentFilePath = "";
    private string crudStatusText = "Ready: use markers 11/12/13/14 to manage context recipes.";
    private string selectedContentRecipeName = "";
    private DateTime lastCrudActionUtc = DateTime.MinValue;
    private int lastCrudActionMarker = -1;

    private System.Windows.Forms.Timer bluetoothTimer;
    private System.Windows.Forms.Timer faceLoginTimer;
    private TcpListener bluetoothServer;
    private Thread bluetoothServerThread;
    private volatile bool bluetoothServerRunning = false;
    private string lastAppliedProfileName = "";
    private List<UserProfile> userProfiles = new List<UserProfile>();
    private UserProfile activeUserProfile = null;
    private HashSet<int> blockedMarkers = new HashSet<int>();

    private string currentUserName = "Guest";
    private string currentWelcomeMessage = "Welcome to Smart Kitchen";
    private string currentBio = "Scan ingredients to get personalized suggestions.";
    private string currentBluetoothDeviceName = "";
    private string currentBluetoothAddress = "";
    private bool isBluetoothSessionActive = false;
    private DateTime lastBluetoothPresenceUtc = DateTime.MinValue;

    // Marker-based face signup state: 60 name, 61 Chef, 62 Client, 63 capture after 8 seconds.
    private string signupStatusText = "Face signup: waiting for unknown face.";
    private string pendingSignupName = "";
    private string pendingSignupRole = "";
    private DateTime lastSignupCommandUtc = DateTime.MinValue;
    private bool signupLaserKeyboardVisible = false;
    private string signupLaserTypedName = "";
    private string signupLaserFocusedKey = "";
    private DateTime signupLaserFocusStartUtc = DateTime.MinValue;
    private DateTime signupLaserLastActionUtc = DateTime.MinValue;
    private const int SIGNUP_LASER_DWELL_MS = 3000;
    private readonly string[] signupKeyboardRows = new string[]
    {
        "QWERTYUIOP",
        "ASDFGHJKL",
        "ZXCVBNM"
    };

    private Process faceRecognitionProcess = null;
    private Process yoloProcess = null;

    //gaze_control
    private TcpClient gazeClient;
    private StreamReader gazeReader;
    private Thread gazeThread;
    private volatile bool gazeRunning = false;

    private const string GAZE_SOCKET_HOST = "127.0.0.1";
    private const int GAZE_SOCKET_PORT = 5001;

    private string lastGazeDirection = "Waiting for gaze tracking...";
    private int gazeLeftHits = 0;
    private int gazeCenterHits = 0;
    private int gazeRightHits = 0;
    private DateTime gazeSessionStartUtc = DateTime.UtcNow;

    // Eye Gaze Heatmap: Marker 50 shows the heatmap overlay on the GUI.
    private const int HEATMAP_MARKER_ID = 50;
    private bool heatmapVisible = false;
    private DateTime lastHeatmapToggleUtc = DateTime.MinValue;
    private List<PointF> heatmapPoints = new List<PointF>();
    private readonly object heatmapSync = new object();
    private Random heatmapRandom = new Random();
    private const int MAX_HEATMAP_POINTS = 1200;
    private string heatmapStatus = "Show marker 50 to display eye gaze heatmap.";

    // Free gaze point: exact point on GUI from Python ScreenX/ScreenY.
    private PointF lastFreeGazePoint = new PointF(0, 0);
    private double lastFreeGazeScreenX = 0.5;
    private double lastFreeGazeScreenY = 0.5;
    private DateTime lastFreeGazeUtc = DateTime.MinValue;

    // Fast gaze zoom: instant visual feedback, while real control remains LEFT/CENTER/RIGHT.
    private MenuView gazeZoomView = MenuView.Overview;
    private DateTime lastGazeZoomUtc = DateTime.MinValue;

    private bool IsGazeZoomActive(MenuView view)
    {
        return gazeZoomView == view &&
               (DateTime.UtcNow - lastGazeZoomUtc).TotalMilliseconds < 1700;
    }

    private void ActivateGazeZoom(MenuView view)
    {
        gazeZoomView = view;
        lastGazeZoomUtc = DateTime.UtcNow;
    }

    private void UpdateFreeGazePoint(float x, float y, double sx, double sy, double confidence)
    {
        lastFreeGazePoint = new PointF(x, y);
        lastFreeGazeScreenX = sx;
        lastFreeGazeScreenY = sy;
        lastFreeGazeUtc = DateTime.UtcNow;

        if (x < WIN_W * 0.34f)
            ActivateGazeZoom(MenuView.Recipes);
        else if (x > WIN_W * 0.66f)
            ActivateGazeZoom(MenuView.Calories);
        else
            ActivateGazeZoom(MenuView.Overview);
    }

    private bool IsFreeGazePointLive()
    {
        return (DateTime.UtcNow - lastFreeGazeUtc).TotalMilliseconds < 900;
    }

    private void ToggleHeatmapByMarker()
    {
        DateTime now = DateTime.UtcNow;
        if ((now - lastHeatmapToggleUtc).TotalMilliseconds < 900)
            return;

        lastHeatmapToggleUtc = now;
        heatmapVisible = !heatmapVisible;
        heatmapStatus = heatmapVisible
            ? "Marker 50 toggled: eye gaze heatmap is visible."
            : "Marker 50 toggled: eye gaze heatmap is hidden.";

        Console.WriteLine(heatmapVisible ? "[HEATMAP] Marker 50 toggle ON." : "[HEATMAP] Marker 50 toggle OFF.");
        Invalidate();
    }
    //
    private enum MenuView
    {
        Overview = 0,
        Recipes = 1,
        Calories = 2,
        Steps = 3,
        Health = 4,
        Tips = 5
    }

    private string[] circularMenuItems = new string[]
    {
        "Overview",
        "Recipes",
        "Calories",
        "Steps",
        "Health",
        "Tips"
    };

    private MenuView currentMenuView = MenuView.Overview;
    private MenuView confirmedMenuView = MenuView.Overview;
    private int lastMenuIndex = -1;
    private DateTime menuHoldStart = DateTime.MinValue;
    private bool menuControllerVisible = false;
    private string menuStatusText = " ";

    // Yellow laser / yellow pen cap control for the circular menu.
    // Python sends: LASER_MENU_INDEX;INDEX=0;PX=0.5;PY=0.5 or LASER_MENU;X=0.5;Y=0.5
    private int laserMenuIndex = -1;
    private DateTime laserMenuHoldStart = DateTime.MinValue;
    private const int LASER_MENU_HOLD_MS = 650;
    private double lastLaserNormX = -1.0;
    private double lastLaserNormY = -1.0;
    private DateTime lastLaserSeenUtc = DateTime.MinValue;
    private string laserStatusText = "Laser: show yellow cap on the circular menu.";
    private int WIN_W = Screen.PrimaryScreen.Bounds.Width;
    private int WIN_H = Screen.PrimaryScreen.Bounds.Height;

    private static Color BG = Color.FromArgb(250, 248, 243);
    private static Color CARD_BG = Color.White;
    private static Color GREEN = Color.FromArgb(29, 158, 117);
    private static Color GREEN_LIGHT = Color.FromArgb(225, 245, 238);
    private static Color AMBER = Color.FromArgb(186, 117, 23);
    private static Color AMBER_LIGHT = Color.FromArgb(250, 238, 218);
    private static Color CORAL = Color.FromArgb(216, 90, 48);
    private static Color CORAL_LIGHT = Color.FromArgb(250, 236, 231);
    private static Color BLUE = Color.FromArgb(52, 120, 246);
    private static Color BLUE_LIGHT = Color.FromArgb(232, 239, 254);
    private static Color PURPLE = Color.FromArgb(121, 84, 181);
    private static Color PURPLE_LIGHT = Color.FromArgb(240, 233, 249);
    private static Color TEAL_DARK = Color.FromArgb(8, 80, 65);
    private static Color GRAY_MID = Color.FromArgb(95, 94, 90);
    private static Color GRAY_LIGHT = Color.FromArgb(241, 239, 232);
    private static Color TEXT_PRI = Color.FromArgb(20, 20, 18);
    private static Color TEXT_SEC = Color.FromArgb(120, 119, 112);

    private Color currentAccent = BLUE;
    private Color currentAccentLight = BLUE_LIGHT;

    private Font fontTitle = new Font("Segoe UI", 22, FontStyle.Bold);
    private Font fontH2 = new Font("Segoe UI", 14, FontStyle.Bold);
    private Font fontH3 = new Font("Segoe UI", 12, FontStyle.Bold);
    private Font fontBody = new Font("Segoe UI", 10);
    private Font fontSmall = new Font("Segoe UI", 9);
    private Font fontBig = new Font("Segoe UI", 25, FontStyle.Bold);
    private Font fontMarker = new Font("Segoe UI", 9, FontStyle.Bold);
    private Font fontTiny = new Font("Segoe UI", 8);
    private Font fontEmoji = new Font("Segoe UI Emoji", 22, FontStyle.Regular);
    /// <summary>
    private const int UNITY_AR_MARKER_ID = 30;
    private Process unityARProcess = null;
    private const string UNITY_AR_EXE = "C:\\Users\\abdo1\\Downloads\\GG\\GG\\HCI Project\\Bulid\\New Unity Project.exe";
    private void StartUnityAR()
    {
        try
        {
            if (unityARProcess != null && !unityARProcess.HasExited)
                return;

            string unityPath = UNITY_AR_EXE;

            if (!Path.IsPathRooted(unityPath))
                unityPath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, unityPath));

            if (!File.Exists(unityPath))
            {
                MessageBox.Show("Unity file not found:\n" + unityPath);
                return;
            }

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = unityPath;
            startInfo.WorkingDirectory = Path.GetDirectoryName(unityPath);
            startInfo.UseShellExecute = true;

            unityARProcess = Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            MessageBox.Show("Unity AR Error:\n" + ex.Message);
        }
    }
    /// </summary>
    private enum ItemType
    {
        Ingredient,
        Tool
    }

    private class UserProfile
    {
        public string SectionName = "";
        public string Name = "";
        public int Age = 0;
        public string BluetoothName = "";
        public string BluetoothAddress = "";
        public string Role = "Chef";
        public List<string> FavoriteMeals = new List<string>();
        public List<string> FavoriteRecipes = new List<string>();
        public string Dislikes = "";
        public HashSet<int> BlockedMarkers = new HashSet<int>();
        public string AccentTheme = "";
        public string WelcomeMessage = "";
        public string Bio = "";
    }

    private class KitchenItem
    {
        public int MarkerID;
        public string Name;
        public string Emoji;
        public ItemType Type;
        public float KcalPerGram;
        public int GramSmall;
        public int GramMed;
        public int GramLarge;
        public int GramXLarge;
        public string[] Tips;
        public Color AccentColor;
        public Color LightColor;

        public bool IsTool
        {
            get { return Type == ItemType.Tool; }
        }

        public int GetGrams(float angle)
        {
            if (IsTool) return 0;

            float deg = angle * 180f / (float)Math.PI;
            if (deg < 0) deg += 360f;

            if (deg < 90f) return GramSmall;
            if (deg < 180f) return GramMed;
            if (deg < 270f) return GramLarge;
            return GramXLarge;
        }

        public string GetQtyLabel(float angle)
        {
            if (IsTool) return "Tool Ready";

            float deg = angle * 180f / (float)Math.PI;
            if (deg < 0) deg += 360f;

            if (deg < 90f) return "Small";
            if (deg < 180f) return "Medium";
            if (deg < 270f) return "Large";
            return "X-Large";
        }

        public int GetCalories(float angle)
        {
            if (IsTool) return 0;
            return (int)(KcalPerGram * GetGrams(angle));
        }
    }
    private class ClientMeal
    {
        public int MarkerID;
        public string Name;
        public string Emoji;
        public string Category;
        public int Calories;
        public int Protein;
        public int Carbs;
        public int Fat;
        public int PrepMinutes;
        public int HealthScore;
        public string Description;
        public string[] Tags;
        public Color AccentColor;
        public Color LightColor;
    }

    /// <summary>
    /// ///////////////////////belal 
    /// </summary>
    private class Recipe
    {
        public string Name;
        public string Emoji;
        public int[] RequiredMarkers;
        public int BaseKcal;
        public string Category;
        public string[] Steps;

        // CRUD Content
        public string Ingredients;
        public string Tools;
        public int Calories;
        public int HealthScore;
        public string Tips;
        public bool IsContentRecipe;
    }
    /////
    private KitchenItem[] items = new KitchenItem[]
    {
        new KitchenItem {
            MarkerID=0, Name="Chicken", Emoji="🍗", Type=ItemType.Ingredient,
            KcalPerGram=1.65f,
            GramSmall=100, GramMed=200, GramLarge=300, GramXLarge=400,
            AccentColor=CORAL, LightColor=CORAL_LIGHT,
            Tips=new string[]{"Use grilled chicken instead of fried.","Rich in protein and suitable for balanced meals.","Remove skin to reduce calories."}
        },
        new KitchenItem {
            MarkerID=1, Name="Tomato", Emoji="🍅", Type=ItemType.Ingredient,
            KcalPerGram=0.18f,
            GramSmall=80, GramMed=150, GramLarge=250, GramXLarge=350,
            AccentColor=GREEN, LightColor=GREEN_LIGHT,
            Tips=new string[]{"Low calorie and rich in antioxidants.","Adds freshness to healthy dishes.","Good source of vitamins."}
        },
        new KitchenItem {
            MarkerID=2, Name="Onion", Emoji="🧅", Type=ItemType.Ingredient,
            KcalPerGram=0.40f,
            GramSmall=50, GramMed=100, GramLarge=150, GramXLarge=200,
            AccentColor=AMBER, LightColor=AMBER_LIGHT,
            Tips=new string[]{"Enhances flavor without much fat.","Useful in soups and many healthy meals."}
        },
        new KitchenItem {
            MarkerID=3, Name="Spices", Emoji="🌶", Type=ItemType.Ingredient,
            KcalPerGram=0.25f,
            GramSmall=5, GramMed=10, GramLarge=20, GramXLarge=30,
            AccentColor=CORAL, LightColor=CORAL_LIGHT,
            Tips=new string[]{"Improves flavor without heavy sauces.","Use herbs and spices to reduce salt.","Turmeric may support anti-inflammatory diets."}
        },
        new KitchenItem {
            MarkerID=4, Name="Rice", Emoji="🍚", Type=ItemType.Ingredient,
            KcalPerGram=1.30f,
            GramSmall=50, GramMed=100, GramLarge=150, GramXLarge=200,
            AccentColor=GRAY_MID, LightColor=GRAY_LIGHT,
            Tips=new string[]{"Control rice portion size.","Brown rice is a healthier option.","Rice gives good energy but watch quantity."}
        },
        new KitchenItem {
            MarkerID=5, Name="Oil", Emoji="💧", Type=ItemType.Ingredient,
            KcalPerGram=8.84f,
            GramSmall=5, GramMed=10, GramLarge=15, GramXLarge=20,
            AccentColor=AMBER, LightColor=AMBER_LIGHT,
            Tips=new string[]{"Use small amounts of oil.","Oil is high in calories.","Prefer olive oil and avoid excess quantity."}
        },
        new KitchenItem {
            MarkerID=6, Name="Spoon", Emoji="🥄", Type=ItemType.Tool,
            KcalPerGram=0f,
            GramSmall=0, GramMed=0, GramLarge=0, GramXLarge=0,
            AccentColor=BLUE, LightColor=BLUE_LIGHT,
            Tips=new string[]{"A spoon is a physical kitchen object used for mixing and serving."}
        },
        new KitchenItem {
            MarkerID=7, Name="Pot", Emoji="🍲", Type=ItemType.Tool,
            KcalPerGram=0f,
            GramSmall=0, GramMed=0, GramLarge=0, GramXLarge=0,
            AccentColor=PURPLE, LightColor=PURPLE_LIGHT,
            Tips=new string[]{"A pot is a physical cooking object required for soups and cooked meals."}
        },
        new KitchenItem {
            MarkerID=8, Name="Knife", Emoji="🔪", Type=ItemType.Tool,
            KcalPerGram=0f,
            GramSmall=0, GramMed=0, GramLarge=0, GramXLarge=0,
            AccentColor=BLUE, LightColor=BLUE_LIGHT,
            Tips=new string[]{"A knife is a physical kitchen object used for chopping and preparation."}
        }
    };

    private ClientMeal[] clientMeals = new ClientMeal[]
    {
        new ClientMeal {
            MarkerID=0, Name="Margherita Pizza", Emoji="🍕", Category="Ready Meal",
            Calories=720, Protein=28, Carbs=86, Fat=28, PrepMinutes=12, HealthScore=64,
            Description="Classic cheese pizza with tomato sauce. Best as a shared meal or a controlled portion.",
            Tags=new string[]{"Popular", "Cheesy", "Fast"}, AccentColor=CORAL, LightColor=CORAL_LIGHT
        },
        new ClientMeal {
            MarkerID=1, Name="Chicken Alfredo Pasta", Emoji="🍝", Category="Pasta Bowl",
            Calories=650, Protein=35, Carbs=72, Fat=24, PrepMinutes=10, HealthScore=70,
            Description="Creamy pasta with chicken strips. Good protein, medium calories if sauce is controlled.",
            Tags=new string[]{"High Protein", "Comfort", "Dinner"}, AccentColor=AMBER, LightColor=AMBER_LIGHT
        },
        new ClientMeal {
            MarkerID=2, Name="Beef Burger Meal", Emoji="🍔", Category="Combo Meal",
            Calories=840, Protein=38, Carbs=78, Fat=42, PrepMinutes=8, HealthScore=52,
            Description="Burger combo suggestion. Choose water and skip extra sauce to reduce calories.",
            Tags=new string[]{"Filling", "Combo", "Treat"}, AccentColor=PURPLE, LightColor=PURPLE_LIGHT
        },
        new ClientMeal {
            MarkerID=3, Name="Sushi Rice Bowl", Emoji="🍣", Category="Light Ready Meal",
            Calories=520, Protein=30, Carbs=68, Fat=14, PrepMinutes=7, HealthScore=82,
            Description="Balanced bowl with rice, fish/chicken, and vegetables. A lighter client option.",
            Tags=new string[]{"Balanced", "Light", "Fresh"}, AccentColor=BLUE, LightColor=BLUE_LIGHT
        },
        new ClientMeal {
            MarkerID=4, Name="Grilled Chicken Wrap", Emoji="🌯", Category="Healthy Fast Meal",
            Calories=480, Protein=40, Carbs=44, Fat=16, PrepMinutes=6, HealthScore=88,
            Description="Lean chicken wrap with vegetables. Best choice for a quick healthy meal.",
            Tags=new string[]{"Best Choice", "Lean", "Quick"}, AccentColor=GREEN, LightColor=GREEN_LIGHT
        },
        new ClientMeal {
            MarkerID=5, Name="Caesar Salad Box", Emoji="🥗", Category="Salad Meal",
            Calories=390, Protein=26, Carbs=22, Fat=20, PrepMinutes=5, HealthScore=84,
            Description="Fresh salad box. Keep dressing on the side for a cleaner nutrition score.",
            Tags=new string[]{"Fresh", "Low Carb", "Healthy"}, AccentColor=TEAL_DARK, LightColor=GREEN_LIGHT
        },
        new ClientMeal {
            MarkerID=6, Name="Breakfast Pancake Box", Emoji="🥞", Category="Breakfast",
            Calories=610, Protein=18, Carbs=88, Fat=20, PrepMinutes=9, HealthScore=58,
            Description="Sweet breakfast option. Better with fruit and less syrup.",
            Tags=new string[]{"Breakfast", "Sweet", "Energy"}, AccentColor=AMBER, LightColor=AMBER_LIGHT
        },
        new ClientMeal {
            MarkerID=7, Name="Fruit Smoothie", Emoji="🥤", Category="Drink",
            Calories=240, Protein=8, Carbs=46, Fat=3, PrepMinutes=3, HealthScore=79,
            Description="Fast drink option. Choose no added sugar for a better score.",
            Tags=new string[]{"Drink", "Fresh", "Quick"}, AccentColor=BLUE, LightColor=BLUE_LIGHT
        },
        new ClientMeal {
            MarkerID=8, Name="Chocolate Dessert Cup", Emoji="🍰", Category="Dessert",
            Calories=430, Protein=6, Carbs=58, Fat=20, PrepMinutes=2, HealthScore=45,
            Description="Dessert recommendation. Enjoy after a light meal, not with a heavy combo.",
            Tags=new string[]{"Dessert", "Sweet", "Small Portion"}, AccentColor=CORAL, LightColor=CORAL_LIGHT
        }
    };

    private Recipe[] recipes = new Recipe[]
    {
        new Recipe {
            Name="Chicken with Tomato",
            Emoji="🍲",
            RequiredMarkers=new int[]{0,1,3,6,7,8},
            BaseKcal=320,
            Category="High Protein",
            Steps=new string[]{
                "Use the knife to cut chicken and tomato.",
                "Place the pot on the cooking area.",
                "Add chicken to the pot.",
                "Add tomato and spices.",
                "Use the spoon to mix and cook for 15 minutes."
            }
        },
        new Recipe {
            Name="Chicken Soup",
            Emoji="🥣",
            RequiredMarkers=new int[]{0,2,3,6,7,8},
            BaseKcal=280,
            Category="Healthy Soup",
            Steps=new string[]{
                "Use the knife to cut onion and chicken.",
                "Place ingredients in the pot.",
                "Add spices and water.",
                "Use the spoon to stir.",
                "Cook until chicken is ready."
            }
        },
        new Recipe {
            Name="Chicken Rice Bowl",
            Emoji="🍛",
            RequiredMarkers=new int[]{0,4,3,5,6,7,8},
            BaseKcal=520,
            Category="Balanced Meal",
            Steps=new string[]{
                "Use the knife to prepare chicken.",
                "Add a small amount of oil to the pot.",
                "Cook chicken with spices.",
                "Add rice and continue cooking.",
                "Serve using the spoon."
            }
        },
        new Recipe {
            Name="Tomato Rice",
            Emoji="🍅",
            RequiredMarkers=new int[]{1,4,5,6,7,8},
            BaseKcal=350,
            Category="Vegetarian",
            Steps=new string[]{
                "Use the knife to cut tomato.",
                "Heat a small amount of oil in the pot.",
                "Add tomato and rice.",
                "Use the spoon to stir.",
                "Cook until rice is soft."
            }
        },
        new Recipe {
            Name="Healthy Chicken Dish",
            Emoji="🥗",
            RequiredMarkers=new int[]{0,1,2,3,6,7,8},
            BaseKcal=410,
            Category="Healthy Balanced Meal",
            Steps=new string[]{
                "Use the knife to prepare chicken, onion, and tomato.",
                "Place everything in the pot.",
                "Add spices only in moderate quantity.",
                "Use the spoon to stir and cook.",
                "Serve as a healthy meal."
            }
        }
    };

    private List<Recipe> contentRecipes = new List<Recipe>();////belallllllll

    private const int PAD = 16;
    private const int CARD_R = 12;

    public SmartKitchenDemo(int port)
    {
        width = WIN_W;
        height = WIN_H;

        this.Text = "Smart Kitchen - TUIO Professional Cooking Assistant";
        this.ClientSize = new Size(WIN_W, WIN_H);
        this.BackColor = BG;
        this.FormBorderStyle = FormBorderStyle.None;
        this.WindowState = FormWindowState.Maximized;
        this.StartPosition = FormStartPosition.CenterScreen;

        this.SetStyle(
            ControlStyles.AllPaintingInWmPaint |
            ControlStyles.UserPaint |
            ControlStyles.DoubleBuffer,
            true
        );

        // No keyboard control is required for CRUD; CRUD is handled by tangible TUIO markers 11-14.
        this.Closing += new CancelEventHandler(Form_Closing);

        ResolveDataPaths();
        LoadProfiles();
        LoadContentRecipes();/////////////////////////belallllllll
        StartBluetoothListener();
        StartBluetoothSocketServer();
        ReadBluetoothProfile();
        StartFaceRecognition();
        StartFaceLoginWatcher();
        StartGestureSocketClient();
        //gaze_control
        StartGazeSocketClient();
        //

        client = new TuioClient(port);
        client.addTuioListener(this);
        client.connect();
    }

    // =========================
    // gaze_control
    // =========================
    private void StartGazeSocketClient()
    {
        gazeRunning = true;
        gazeThread = new Thread(GazeSocketLoop);
        gazeThread.IsBackground = true;
        gazeThread.Start();
    }

    private void GazeSocketLoop()
    {
        while (gazeRunning)
        {
            try
            {
                if (gazeClient != null)
                {
                    try { gazeClient.Close(); } catch { }
                    gazeClient = null;
                }

                Console.WriteLine("[GAZE] Connecting to Python on " + GAZE_SOCKET_HOST + ":" + GAZE_SOCKET_PORT);
                gazeClient = new TcpClient();
                gazeClient.Connect(GAZE_SOCKET_HOST, GAZE_SOCKET_PORT);
                gazeReader = new StreamReader(gazeClient.GetStream());

                lastGazeDirection = "Gaze tracking connected.";
                SafeInvalidate();

                while (gazeRunning && gazeClient.Connected)
                {
                    string line = gazeReader.ReadLine();
                    if (line == null) break;

                    string command = line.Trim();
                    if (command.Length == 0) continue;

                    HandleGazeCommand(command);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("[GAZE SOCKET] " + ex.Message);
                lastGazeDirection = "Waiting for Python gaze server...";
                SafeInvalidate();
            }

            if (!gazeRunning) break;
            Thread.Sleep(1500);
        }
    }

    private void HandleGazeCommand(string command)
    {
        if (this.IsDisposed) return;

        if (this.InvokeRequired)
        {
            this.BeginInvoke((MethodInvoker)delegate { HandleGazeCommand(command); });
            return;
        }

        if (!isLoggedIn)
        {
            lastGazeDirection = "Gaze detected but user is not logged in.";
            Invalidate();
            return;
        }

        if (!isLoggedIn)
        {
            lastGazeDirection = "Gaze detected but user is not logged in.";
            Invalidate();
            return;
        }

        string[] parts = command.Split(';');
        string mainCommand = parts[0].Trim().ToUpperInvariant();
        Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (string part in parts)
        {
            int idx = part.IndexOf('=');
            if (idx > 0)
            {
                string key = part.Substring(0, idx).Trim();
                string value = part.Substring(idx + 1).Trim();
                values[key] = value;
            }
        }

        AddHeatmapPointFromGaze(mainCommand, values);

        // GAZE_SAMPLE gives instant zoom only. It does NOT change the selected menu/control.
        if (mainCommand == "GAZE_SAMPLE")
        {
            string dir = values.ContainsKey("Direction") ? values["Direction"].Trim().ToUpperInvariant() : "";
            string zone = values.ContainsKey("Zone") ? values["Zone"].Trim().ToUpperInvariant() : "";

            string label = dir.Replace("_", " ");
            string zoneLabel = zone.Replace("_", " ");

            if (values.ContainsKey("ScreenX") && values.ContainsKey("ScreenY"))
            {
                if (zoneLabel.Length > 0)
                    lastGazeDirection = "Free gaze: " + label + "  |  3-Zone Focus: " + zoneLabel + "  X=" + values["ScreenX"] + " Y=" + values["ScreenY"];
                else
                    lastGazeDirection = "Free gaze: " + label + "  X=" + values["ScreenX"] + " Y=" + values["ScreenY"];
            }
            else
            {
                lastGazeDirection = "Looking: " + label;
            }

            Invalidate();
            return;
        }

        // Stable commands: these are the only commands that control the GUI.
        if (mainCommand == "GAZE_LEFT")
        {
            gazeLeftHits++;
            ActivateGazeZoom(MenuView.Recipes);
            lastGazeDirection = "Looking LEFT";
        }
        else if (mainCommand == "GAZE_CENTER")
        {
            gazeCenterHits++;
            ActivateGazeZoom(MenuView.Overview);
            lastGazeDirection = "Looking CENTER";
        }
        else if (mainCommand == "GAZE_RIGHT")
        {
            gazeRightHits++;
            ActivateGazeZoom(MenuView.Calories);
            lastGazeDirection = "Looking RIGHT";
        }
        else if (mainCommand.StartsWith("GAZE_") || mainCommand == "GAZE_POINT")
        {
            string label = mainCommand.Replace("GAZE_", "").Replace("_", " ");

            if (values.ContainsKey("Direction"))
                label = values["Direction"].Replace("_", " ");

            lastGazeDirection = "Looking: " + label;
        }
        else
        {
            lastGazeDirection = "Unknown gaze command: " + command;
        }
        Invalidate();
    }
    //
    // =========================
    // Bluetooth + Profiles
    // =========================
    private void ResolveDataPaths()
    {
        profileFilePath = ResolveFilePath(PROFILE_FILE);
        activeBluetoothFilePath = ResolveFilePath(ACTIVE_BT_FILE);
        activeFaceFilePath = ResolveFilePath(FACE_LOGIN_FILE);
        contentFilePath = ResolveFilePath("SmartKitchenContent.txt");

        if (string.IsNullOrWhiteSpace(profileFilePath))
            profileFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, PROFILE_FILE);

        if (string.IsNullOrWhiteSpace(activeBluetoothFilePath))
            activeBluetoothFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ACTIVE_BT_FILE);

        if (string.IsNullOrWhiteSpace(activeFaceFilePath))
            activeFaceFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, FACE_LOGIN_FILE);

        Console.WriteLine("[PATH] Profiles file: " + profileFilePath);
        Console.WriteLine("[PATH] Active Bluetooth file: " + activeBluetoothFilePath);
        Console.WriteLine("[PATH] Active face login file: " + activeFaceFilePath);
    }

    private string ResolveFilePath(string fileName)
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string startupDir = System.Windows.Forms.Application.StartupPath;
        string currentDir = Directory.GetCurrentDirectory();

        List<string> candidates = new List<string>();
        candidates.Add(Path.Combine(baseDir, fileName));
        candidates.Add(Path.Combine(startupDir, fileName));
        candidates.Add(Path.Combine(currentDir, fileName));

        DirectoryInfo dir = new DirectoryInfo(baseDir);
        for (int i = 0; i < 6 && dir != null; i++, dir = dir.Parent)
            candidates.Add(Path.Combine(dir.FullName, fileName));

        foreach (string path in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                if (File.Exists(path)) return path;
            }
            catch { }
        }

        return Path.Combine(baseDir, fileName);
    }

    private void LoadProfiles()
    {
        userProfiles.Clear();

        if (!File.Exists(profileFilePath))
        {
            Console.WriteLine("[PROFILE FILE NOT FOUND] " + profileFilePath);
            return;
        }

        string[] lines = File.ReadAllLines(profileFilePath);
        UserProfile current = null;

        foreach (string raw in lines)
        {
            string line = raw.Trim();
            if (line == "" || line.StartsWith("#")) continue;

            if (line.StartsWith("[") && line.EndsWith("]"))
            {
                current = new UserProfile();
                current.SectionName = line.Substring(1, line.Length - 2).Trim();
                userProfiles.Add(current);
                continue;
            }

            if (current == null) continue;

            int idx = line.IndexOf('=');
            if (idx <= 0) continue;

            string key = line.Substring(0, idx).Trim().ToLower();
            string value = line.Substring(idx + 1).Trim();

            switch (key)
            {
                case "name":
                    current.Name = value;
                    break;
                case "age":
                    int age;
                    if (int.TryParse(value, out age)) current.Age = age;
                    break;
                case "role":
                    current.Role = string.IsNullOrWhiteSpace(value) ? "Chef" : NormalizeRole(value);
                    break;
                case "bluetoothname":
                    current.BluetoothName = value;
                    break;
                case "bluetoothaddress":
                    current.BluetoothAddress = value;
                    break;
                case "favoritemeals":
                    current.FavoriteMeals = SplitCsv(value);
                    break;
                case "favoriterecipes":
                    current.FavoriteRecipes = SplitCsv(value);
                    break;
                case "dislikes":
                    current.Dislikes = value;
                    break;
                case "blockedmarkers":
                    current.BlockedMarkers = ParseBlockedMarkers(value);
                    break;
                case "accenttheme":
                    current.AccentTheme = value;
                    break;
                case "welcomemessage":
                    current.WelcomeMessage = value;
                    break;
                case "bio":
                    current.Bio = value;
                    break;
            }
        }
    }



    // =========================
    // CRUD Content based on the context - Belal
    // READ / CREATE / UPDATE / DELETE recipes from SmartKitchenContent.txt
    // The content is not a fake fixed card; it is connected to scanned markers.
    // =========================
    private void LoadContentRecipes()
    {
        contentRecipes.Clear();

        if (string.IsNullOrWhiteSpace(contentFilePath))
            contentFilePath = ResolveFilePath("SmartKitchenContent.txt");

        if (string.IsNullOrWhiteSpace(contentFilePath))
            contentFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SmartKitchenContent.txt");

        if (!File.Exists(contentFilePath))
        {
            SaveContentRecipes();
            crudStatusText = "Content file created. Place marker 11 to add from scanned context.";
            return;
        }

        string[] lines = File.ReadAllLines(contentFilePath, Encoding.UTF8);
        Recipe current = null;

        foreach (string raw in lines)
        {
            string line = raw.Trim();

            if (line.Length == 0 || line.StartsWith("#"))
                continue;

            if (line.StartsWith("[") && line.EndsWith("]"))
            {
                AddLoadedContentRecipe(current);
                current = new Recipe();
                current.IsContentRecipe = true;
                current.Emoji = "📄";
                current.Category = "Context Content";
                current.Steps = new string[0];
                continue;
            }

            if (current == null || !line.Contains("="))
                continue;

            int idx = line.IndexOf('=');
            string key = line.Substring(0, idx).Trim().ToLowerInvariant();
            string value = line.Substring(idx + 1).Trim();

            switch (key)
            {
                case "name":
                    current.Name = value;
                    break;
                case "category":
                    current.Category = value;
                    break;
                case "ingredients":
                    current.Ingredients = value;
                    break;
                case "tools":
                    current.Tools = value;
                    break;
                case "calories":
                    int cal;
                    if (int.TryParse(value, out cal)) current.Calories = cal;
                    break;
                case "healthscore":
                    int score;
                    if (int.TryParse(value, out score)) current.HealthScore = score;
                    break;
                case "tips":
                    current.Tips = value;
                    break;
                case "steps":
                    current.Steps = SplitCsv(value).ToArray();
                    break;
            }
        }

        AddLoadedContentRecipe(current);
        if (GetSelectedContentRecipe() == null) selectedContentRecipeName = "";
        crudStatusText = "📂 Loaded " + contentRecipes.Count + " context recipes from file.";
    }

    private void AddLoadedContentRecipe(Recipe recipe)
    {
        if (recipe == null) return;
        if (string.IsNullOrWhiteSpace(recipe.Name)) return;

        recipe.IsContentRecipe = true;
        recipe.Emoji = string.IsNullOrWhiteSpace(recipe.Emoji) ? "📄" : recipe.Emoji;
        recipe.RequiredMarkers = GuessMarkersFromContent(recipe.Ingredients, recipe.Tools);
        recipe.BaseKcal = recipe.Calories;

        if (recipe.HealthScore <= 0) recipe.HealthScore = 70;
        if (string.IsNullOrWhiteSpace(recipe.Tips))
            recipe.Tips = "Follow the healthy cooking guidance for the scanned context.";

        if (recipe.Steps == null || recipe.Steps.Length == 0)
            recipe.Steps = BuildContentRecipeSteps(recipe);

        contentRecipes.Add(recipe);
    }

    private string EscapeContentValue(string value)
    {
        return (value ?? "").Replace(Environment.NewLine, " ").Replace("\r", " ").Replace("\n", " ").Trim();
    }

    private void SaveContentRecipes()
    {
        if (string.IsNullOrWhiteSpace(contentFilePath))
            contentFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "SmartKitchenContent.txt");

        StringBuilder sb = new StringBuilder();
        sb.AppendLine("# Smart Kitchen context CRUD content");
        sb.AppendLine("# Tangible markers: M11=Create, M12=Update, M13=Delete, M14=Read/Reload");
        sb.AppendLine("# Each recipe is matched with the scanned TUIO ingredient/tool markers.");
        sb.AppendLine();

        for (int i = 0; i < contentRecipes.Count; i++)
        {
            Recipe r = contentRecipes[i];
            sb.AppendLine("[Recipe" + (i + 1).ToString() + "]");
            sb.AppendLine("Name=" + EscapeContentValue(r.Name));
            sb.AppendLine("Category=" + EscapeContentValue(r.Category));
            sb.AppendLine("Ingredients=" + EscapeContentValue(r.Ingredients));
            sb.AppendLine("Tools=" + EscapeContentValue(r.Tools));
            sb.AppendLine("Calories=" + r.Calories.ToString());
            sb.AppendLine("HealthScore=" + r.HealthScore.ToString());
            sb.AppendLine("Tips=" + EscapeContentValue(r.Tips));
            if (r.Steps != null && r.Steps.Length > 0)
                sb.AppendLine("Steps=" + EscapeContentValue(string.Join(",", r.Steps)));
            sb.AppendLine();
        }

        File.WriteAllText(contentFilePath, sb.ToString(), Encoding.UTF8);
    }

    private string[] BuildContentRecipeSteps(Recipe recipe)
    {
        List<string> steps = new List<string>();
        steps.Add("Prepare scanned ingredients: " + (string.IsNullOrWhiteSpace(recipe.Ingredients) ? "no ingredients" : recipe.Ingredients) + ".");
        steps.Add("Use available tools: " + (string.IsNullOrWhiteSpace(recipe.Tools) ? "no tools" : recipe.Tools) + ".");
        steps.Add(string.IsNullOrWhiteSpace(recipe.Tips) ? "Follow the healthy cooking recommendation." : recipe.Tips);
        return steps.ToArray();
    }

    private List<KitchenItem> GetDetectedKitchenItems(Dictionary<int, TuioDemoObject> detected, bool tools)
    {
        List<KitchenItem> result = new List<KitchenItem>();
        foreach (int id in detected.Keys)
        {
            KitchenItem item = items.FirstOrDefault(i => i.MarkerID == id);
            if (item == null) continue;
            if (tools && item.IsTool) result.Add(item);
            if (!tools && !item.IsTool) result.Add(item);
        }
        return result.OrderBy(i => i.MarkerID).ToList();
    }

    private string BuildContextTip(Dictionary<int, TuioDemoObject> detected)
    {
        int score = CalculateHealthScore(detected);
        int calories = GetTotalCalories(detected);

        if (detected.ContainsKey(5) && calories > 700)
            return "Use less oil and reduce the portion size for a healthier meal.";
        if (detected.ContainsKey(0) && detected.ContainsKey(4) && !detected.ContainsKey(1))
            return "Add tomato or vegetables to balance chicken and rice.";
        if (score >= 85)
            return "This is a healthy context. Keep oil moderate and use fresh ingredients.";
        if (score >= 70)
            return "Good meal choice. Improve it by adding vegetables and controlling oil.";
        return "Needs improvement: reduce high calorie ingredients and add vegetables.";
    }

    private Recipe BuildRecipeFromCurrentContext()
    {
        Dictionary<int, TuioDemoObject> detected = GetFilteredDetected();
        List<KitchenItem> ingredientItems = GetDetectedKitchenItems(detected, false);
        List<KitchenItem> toolItems = GetDetectedKitchenItems(detected, true);

        if (ingredientItems.Count == 0)
            return null;

        string ingredients = string.Join(",", ingredientItems.Select(i => i.Name));
        string tools = string.Join(",", toolItems.Select(i => i.Name));
        int calories = GetTotalCalories(detected);
        int health = CalculateHealthScore(detected);
        string mainName = string.Join(" ", ingredientItems.Take(3).Select(i => i.Name));

        Recipe recipe = new Recipe();
        recipe.Name = "Context " + mainName + " Recipe";
        recipe.Category = activeUserProfile != null ? "Context for " + activeUserProfile.Name : "Context Recipe";
        recipe.Ingredients = ingredients;
        recipe.Tools = tools;
        recipe.Calories = calories;
        recipe.HealthScore = health;
        recipe.Tips = BuildContextTip(detected);
        recipe.RequiredMarkers = detected.Keys.Where(id => items.Any(i => i.MarkerID == id)).OrderBy(id => id).ToArray();
        recipe.BaseKcal = calories;
        recipe.Emoji = "📄";
        recipe.IsContentRecipe = true;
        recipe.Steps = BuildContentRecipeSteps(recipe);
        return recipe;
    }

    private int[] GetCurrentContextMarkerIds()
    {
        Dictionary<int, TuioDemoObject> detected = GetFilteredDetected();
        return detected.Keys
            .Where(id => items.Any(i => i.MarkerID == id) && !IsCrudMarker(id) && id != MENU_MARKER_ID && id != LOGIN_MARKER_ID)
            .Distinct()
            .OrderBy(id => id)
            .ToArray();
    }

    private string BuildMarkerSignature(IEnumerable<int> markers)
    {
        if (markers == null) return "";
        return string.Join("|", markers.Distinct().OrderBy(id => id).Select(id => id.ToString()).ToArray());
    }

    private string GetRecipeSignature(Recipe recipe)
    {
        if (recipe == null) return "";
        if (recipe.RequiredMarkers == null || recipe.RequiredMarkers.Length == 0)
            recipe.RequiredMarkers = GuessMarkersFromContent(recipe.Ingredients, recipe.Tools);
        return BuildMarkerSignature(recipe.RequiredMarkers);
    }

    private Recipe FindExactContentRecipeForCurrentContext()
    {
        string sig = BuildMarkerSignature(GetCurrentContextMarkerIds());
        if (string.IsNullOrWhiteSpace(sig)) return null;

        return contentRecipes.FirstOrDefault(r => string.Equals(GetRecipeSignature(r), sig, StringComparison.OrdinalIgnoreCase));
    }

    private Recipe GetSelectedContentRecipe()
    {
        if (string.IsNullOrWhiteSpace(selectedContentRecipeName)) return null;
        return contentRecipes.FirstOrDefault(r => string.Equals(r.Name, selectedContentRecipeName, StringComparison.OrdinalIgnoreCase));
    }

    private void SelectContentRecipe(Recipe recipe)
    {
        selectedContentRecipeName = recipe == null ? "" : recipe.Name;
    }

    private Recipe FindTargetContentRecipe()
    {
        Recipe exact = FindExactContentRecipeForCurrentContext();
        if (exact != null) return exact;

        Recipe selected = GetSelectedContentRecipe();
        if (selected != null) return selected;

        return FindBestContentRecipeForContext();
    }

    private Recipe FindBestContentRecipeForContext()
    {
        if (contentRecipes == null || contentRecipes.Count == 0) return null;

        Dictionary<int, TuioDemoObject> detected = GetFilteredDetected();
        HashSet<int> active = new HashSet<int>(detected.Keys);

        Recipe best = null;
        int bestScore = -1;

        foreach (Recipe r in contentRecipes)
        {
            int overlap = 0;
            if (r.RequiredMarkers != null)
            {
                foreach (int id in r.RequiredMarkers)
                {
                    if (active.Contains(id)) overlap++;
                }
            }

            int score = overlap * 10 - Math.Abs((r.RequiredMarkers == null ? 0 : r.RequiredMarkers.Length) - active.Count);
            if (score > bestScore)
            {
                bestScore = score;
                best = r;
            }
        }

        return best;
    }

    private void CreateContextRecipe()
    {
        Recipe recipe = BuildRecipeFromCurrentContext();
        if (recipe == null)
        {
            crudStatusText = "⚠ Add failed: scan at least one ingredient first, then place M11.";
            Invalidate();
            return;
        }

        Recipe duplicate = FindExactContentRecipeForCurrentContext();
        if (duplicate != null)
        {
            SelectContentRecipe(duplicate);
            crudStatusText = "⚠ Already exists: " + duplicate.Name + " matches the current context. Use M12 to update it.";
            Invalidate();
            return;
        }

        string baseName = recipe.Name;
        int n = 2;
        while (contentRecipes.Any(r => string.Equals(r.Name, recipe.Name, StringComparison.OrdinalIgnoreCase)))
        {
            recipe.Name = baseName + " " + n.ToString();
            n++;
        }

        contentRecipes.Add(recipe);
        SelectContentRecipe(recipe);
        SaveContentRecipes();
        crudStatusText = "✅ Added: " + recipe.Name + " saved from the current scanned context.";
        Invalidate();
    }

    private void UpdateContextRecipe()
    {
        Recipe target = FindTargetContentRecipe();
        Recipe fresh = BuildRecipeFromCurrentContext();

        if (target == null)
        {
            crudStatusText = "⚠ Update failed: no context recipe exists. Use M11 to create one first.";
            Invalidate();
            return;
        }

        if (fresh == null)
        {
            target.Tips = "Updated at " + DateTime.Now.ToString("HH:mm") + ": scan ingredients/tools to refresh this recipe context.";
            SelectContentRecipe(target);
            SaveContentRecipes();
            crudStatusText = "🔄 Updated tip only: " + target.Name + ".";
            Invalidate();
            return;
        }

        Recipe duplicate = FindExactContentRecipeForCurrentContext();
        if (duplicate != null && !object.ReferenceEquals(duplicate, target))
        {
            SelectContentRecipe(duplicate);
            crudStatusText = "⚠ Update skipped: current context already belongs to " + duplicate.Name + ".";
            Invalidate();
            return;
        }

        target.Ingredients = fresh.Ingredients;
        target.Tools = fresh.Tools;
        target.Calories = fresh.Calories;
        target.HealthScore = fresh.HealthScore;
        target.Tips = fresh.Tips;
        target.RequiredMarkers = fresh.RequiredMarkers;
        target.BaseKcal = fresh.BaseKcal;
        target.Steps = BuildContentRecipeSteps(target);

        SelectContentRecipe(target);
        SaveContentRecipes();
        crudStatusText = "🔄 Updated: " + target.Name + " now matches the current scanned context.";
        Invalidate();
    }

    private void DeleteContextRecipe()
    {
        Recipe target = FindTargetContentRecipe();
        if (target == null)
        {
            crudStatusText = "⚠ Delete failed: no context recipe is available to remove.";
            Invalidate();
            return;
        }

        string deletedName = target.Name;
        contentRecipes.Remove(target);
        if (string.Equals(selectedContentRecipeName, deletedName, StringComparison.OrdinalIgnoreCase))
            selectedContentRecipeName = "";
        SaveContentRecipes();
        crudStatusText = "🗑 Deleted: " + deletedName + " removed from context content.";
        Invalidate();
    }

    private bool IsCrudMarker(int markerId)
    {
        return markerId == CRUD_ADD_MARKER_ID ||
               markerId == CRUD_UPDATE_MARKER_ID ||
               markerId == CRUD_DELETE_MARKER_ID ||
               markerId == CRUD_RELOAD_MARKER_ID;
    }

    private void HandleCrudMarker(int markerId)
    {
        DateTime now = DateTime.UtcNow;
        if (lastCrudActionMarker == markerId &&
            (now - lastCrudActionUtc).TotalMilliseconds < CRUD_ACTION_COOLDOWN_MS)
            return;

        lastCrudActionMarker = markerId;
        lastCrudActionUtc = now;

        if (markerId == CRUD_ADD_MARKER_ID)
        {
            Console.WriteLine("[CRUD MARKER] M11 Add context recipe");
            CreateContextRecipe();
        }
        else if (markerId == CRUD_UPDATE_MARKER_ID)
        {
            Console.WriteLine("[CRUD MARKER] M12 Update context recipe");
            UpdateContextRecipe();
        }
        else if (markerId == CRUD_DELETE_MARKER_ID)
        {
            Console.WriteLine("[CRUD MARKER] M13 Delete context recipe");
            DeleteContextRecipe();
        }
        else if (markerId == CRUD_RELOAD_MARKER_ID)
        {
            Console.WriteLine("[CRUD MARKER] M14 Reload content file");
            LoadContentRecipes();
            Invalidate();
        }
    }

    private List<string> SplitCsv(string value)
    {
        return value
            .Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(x => x.Trim())
            .Where(x => x.Length > 0)
            .ToList();
    }

    private HashSet<int> ParseBlockedMarkers(string value)
    {
        HashSet<int> set = new HashSet<int>();
        foreach (string s in value.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
        {
            int x;
            if (int.TryParse(s.Trim(), out x))
                set.Add(x);
        }
        return set;
    }

    private void StartBluetoothListener()
    {
        bluetoothTimer = new System.Windows.Forms.Timer();
        bluetoothTimer.Interval = 1200;
        bluetoothTimer.Tick += (s, e) =>
        {
            ReadBluetoothProfile();
            CheckBluetoothPresenceTimeout();
        };
        bluetoothTimer.Start();
    }

    private void CheckBluetoothPresenceTimeout()
    {
        if (!isBluetoothSessionActive)
            return;

        if (lastBluetoothPresenceUtc == DateTime.MinValue)
            return;

        TimeSpan elapsed = DateTime.UtcNow - lastBluetoothPresenceUtc;
        if (elapsed.TotalSeconds >= BLUETOOTH_PRESENCE_TIMEOUT_SECONDS)
        {
            Console.WriteLine("[BLUETOOTH TIMEOUT] Device is no longer present.");
            LogoutToLogin("Bluetooth device is no longer present.");
        }
    }

    private void StartBluetoothSocketServer()
    {
        try
        {
            bluetoothServer = new TcpListener(IPAddress.Parse(SOCKET_HOST), SOCKET_PORT);
            bluetoothServer.Start();
            bluetoothServerRunning = true;

            bluetoothServerThread = new Thread(ListenForBluetoothClients);
            bluetoothServerThread.IsBackground = true;
            bluetoothServerThread.Start();

            Console.WriteLine("[SOCKET] Bluetooth listener started on " + SOCKET_HOST + ":" + SOCKET_PORT);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[SOCKET ERROR] Could not start Bluetooth listener: " + ex.Message);
        }
    }

    private void ListenForBluetoothClients()
    {
        while (bluetoothServerRunning)
        {
            TcpClient socketClient = null;

            try
            {
                socketClient = bluetoothServer.AcceptTcpClient();

                using (socketClient)
                using (NetworkStream stream = socketClient.GetStream())
                using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
                using (StreamWriter writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true })
                {
                    string message = reader.ReadLine();

                    if (string.IsNullOrWhiteSpace(message))
                    {
                        writer.WriteLine("EMPTY");
                        continue;
                    }

                    Console.WriteLine("[SOCKET RECEIVED] " + message);
                    HandleBluetoothSocketMessage(message);
                    writer.WriteLine("OK");
                }
            }
            catch (SocketException)
            {
                if (!bluetoothServerRunning)
                    break;
            }
            catch (Exception ex)
            {
                Console.WriteLine("[SOCKET CLIENT ERROR] " + ex.Message);
            }
        }
    }


    private void SendSignupCommandToPython(string command)
    {
        try
        {
            using (TcpClient c = new TcpClient())
            {
                c.Connect(SOCKET_HOST, SIGNUP_COMMAND_PORT);
                using (StreamWriter sw = new StreamWriter(c.GetStream()))
                {
                    sw.AutoFlush = true;
                    sw.WriteLine(command);
                }
            }
            lastSignupCommandUtc = DateTime.UtcNow;
            Console.WriteLine("[SIGNUP COMMAND SENT] " + command);
        }
        catch (Exception ex)
        {
            signupStatusText = "Python signup socket is not ready. Run smart_kitchen_all_in_one_camera.py first.";
            Console.WriteLine("[SIGNUP COMMAND ERROR] " + ex.Message);
            Invalidate();
        }
    }

    private string SafeSignupValue(string value)
    {
        return (value ?? "").Replace(";", " ").Replace("\r", " ").Replace("\n", " ").Trim();
    }


    private bool CanRunSignupMarker()
    {
        return (DateTime.UtcNow - lastSignupCommandUtc).TotalMilliseconds > 1500;
    }

    private void TriggerSignupMarkerFromTuio(int markerId)
    {
        if (this.IsDisposed) return;

        if (this.InvokeRequired)
        {
            this.BeginInvoke((MethodInvoker)delegate
            {
                TriggerSignupMarkerFromTuio(markerId);
            });
            return;
        }

        if (!CanRunSignupMarker())
            return;

        lastSignupCommandUtc = DateTime.UtcNow;
        Console.WriteLine("[SIGNUP MARKER] Marker " + markerId + " triggered.");
        HandleSignupMarker(markerId);
    }

    private void HandleSignupMarker(int markerId)
    {
        if (markerId == SIGNUP_NAME_MARKER_ID)
        {
            OpenSignupLaserKeyboard();
        }
        else if (markerId == SIGNUP_CHEF_MARKER_ID || markerId == SIGNUP_CLIENT_MARKER_ID)
        {
            if (pendingSignupName.Length == 0)
            {
                signupStatusText = "Use marker 60 first to enter the user name.";
                Invalidate();
                return;
            }

            pendingSignupRole = markerId == SIGNUP_CHEF_MARKER_ID ? "Chef" : "Client";
            signupStatusText = pendingSignupRole + " selected. Use marker 63 to start 8 second capture countdown.";
            SendSignupCommandToPython("SIGNUP_ROLE;Role=" + pendingSignupRole);
        }
        else if (markerId == SIGNUP_CAPTURE_MARKER_ID)
        {
            if (pendingSignupName.Length == 0)
                signupStatusText = "Use marker 60 first to enter the user name.";
            else if (pendingSignupRole.Length == 0)
                signupStatusText = "Use marker 61 for Chef or marker 62 for Client before capture.";
            else
            {
                signupStatusText = "Countdown started: 8 seconds. Look at the camera and hold still.";
                SendSignupCommandToPython("SIGNUP_CAPTURE;Seconds=8");
            }
        }

        Invalidate();
    }


    private void OpenSignupLaserKeyboard()
    {
        signupLaserKeyboardVisible = true;
        signupLaserTypedName = "";
        signupLaserFocusedKey = "";
        signupLaserFocusStartUtc = DateTime.MinValue;
        signupStatusText = "Laser keyboard opened. Point the yellow laser at letters and hold 3 seconds to type.";
        laserStatusText = "Signup keyboard: hold laser on a key for 3 seconds to choose it.";
        Invalidate();
    }

    private Rectangle GetSignupKeyboardBounds()
    {
        int w = Math.Min(1120, WIN_W - 140);
        int h = 390;
        int x = (WIN_W - w) / 2;
        int y = WIN_H - h - 44;
        if (y < 250) y = 250;
        return new Rectangle(x, y, w, h);
    }

    private List<KeyValuePair<string, Rectangle>> GetSignupKeyboardKeys()
    {
        Rectangle box = GetSignupKeyboardBounds();
        List<KeyValuePair<string, Rectangle>> keys = new List<KeyValuePair<string, Rectangle>>();

        int keyW = 78;
        int keyH = 56;
        int gap = 10;
        int y = box.Y + 112;

        for (int r = 0; r < signupKeyboardRows.Length; r++)
        {
            string row = signupKeyboardRows[r];
            int rowW = row.Length * keyW + (row.Length - 1) * gap;
            int x = box.X + (box.Width - rowW) / 2 + (r == 1 ? 0 : 0);

            for (int i = 0; i < row.Length; i++)
            {
                string key = row[i].ToString();
                keys.Add(new KeyValuePair<string, Rectangle>(key, new Rectangle(x + i * (keyW + gap), y + r * (keyH + gap), keyW, keyH)));
            }
        }

        int bottomY = box.Y + 318;
        int x0 = box.X + 150;
        keys.Add(new KeyValuePair<string, Rectangle>("SPACE", new Rectangle(x0, bottomY, 220, 54)));
        keys.Add(new KeyValuePair<string, Rectangle>("BACK", new Rectangle(x0 + 235, bottomY, 130, 54)));
        keys.Add(new KeyValuePair<string, Rectangle>("CLEAR", new Rectangle(x0 + 380, bottomY, 130, 54)));
        keys.Add(new KeyValuePair<string, Rectangle>("OK", new Rectangle(x0 + 525, bottomY, 130, 54)));
        keys.Add(new KeyValuePair<string, Rectangle>("CANCEL", new Rectangle(x0 + 670, bottomY, 150, 54)));

        return keys;
    }

    private string HitSignupKeyboardKey(int x, int y)
    {
        foreach (KeyValuePair<string, Rectangle> kv in GetSignupKeyboardKeys())
        {
            if (kv.Value.Contains(x, y))
                return kv.Key;
        }
        return "";
    }

    private void ProcessSignupLaserKeyboard(double nx, double ny)
    {
        if (!signupLaserKeyboardVisible)
            return;

        int x = (int)(nx * WIN_W);
        int y = (int)(ny * WIN_H);
        string key = HitSignupKeyboardKey(x, y);

        if (key.Length == 0)
        {
            signupLaserFocusedKey = "";
            signupLaserFocusStartUtc = DateTime.MinValue;
            laserStatusText = "Signup keyboard: point at a key.";
            Invalidate();
            return;
        }

        DateTime now = DateTime.UtcNow;

        if (key != signupLaserFocusedKey)
        {
            signupLaserFocusedKey = key;
            signupLaserFocusStartUtc = now;
            laserStatusText = "Signup keyboard focus: " + key + " • hold 3 sec to select.";
            Invalidate();
            return;
        }

        if ((now - signupLaserLastActionUtc).TotalMilliseconds < 350)
        {
            Invalidate();
            return;
        }

        if ((now - signupLaserFocusStartUtc).TotalMilliseconds >= SIGNUP_LASER_DWELL_MS)
        {
            ApplySignupKeyboardKey(key);
            signupLaserLastActionUtc = now;
            signupLaserFocusStartUtc = now;
            Invalidate();
        }
    }

    private void ApplySignupKeyboardKey(string key)
    {
        if (key.Length == 1 && char.IsLetter(key[0]))
        {
            if (signupLaserTypedName.Length < 24)
                signupLaserTypedName += key;
        }
        else if (key == "SPACE")
        {
            if (signupLaserTypedName.Length > 0 && signupLaserTypedName.Length < 24 && !signupLaserTypedName.EndsWith(" "))
                signupLaserTypedName += " ";
        }
        else if (key == "BACK")
        {
            if (signupLaserTypedName.Length > 0)
                signupLaserTypedName = signupLaserTypedName.Substring(0, signupLaserTypedName.Length - 1);
        }
        else if (key == "CLEAR")
        {
            signupLaserTypedName = "";
        }
        else if (key == "CANCEL")
        {
            signupLaserKeyboardVisible = false;
            signupLaserTypedName = "";
            signupLaserFocusedKey = "";
            signupStatusText = "Signup cancelled. Show marker 60 again to enter a name.";
            return;
        }
        else if (key == "OK")
        {
            string finalName = signupLaserTypedName.Trim();
            if (finalName.Length == 0)
            {
                signupStatusText = "Name is empty. Use the laser keyboard to type a valid name.";
                return;
            }

            pendingSignupName = finalName;
            pendingSignupRole = "";
            signupLaserKeyboardVisible = false;
            signupLaserFocusedKey = "";
            signupStatusText = "Name saved: " + pendingSignupName + ". Use marker 61 for Chef or marker 62 for Client.";
            SendSignupCommandToPython("SIGNUP_START;Name=" + SafeSignupValue(pendingSignupName));
            return;
        }

        signupStatusText = "Typing name: " + (signupLaserTypedName.Trim().Length == 0 ? "_" : signupLaserTypedName);
    }

    private void DrawSignupLaserKeyboard(Graphics g)
    {
        if (!signupLaserKeyboardVisible)
            return;

        Rectangle box = GetSignupKeyboardBounds();

        using (SolidBrush dim = new SolidBrush(Color.FromArgb(125, 0, 0, 0)))
            g.FillRectangle(dim, new Rectangle(0, 0, WIN_W, WIN_H));

        FillRoundRect(g, Color.FromArgb(252, 250, 244), box, 28);
        DrawRoundRect(g, Color.FromArgb(80, 0, 0, 0), box, 28);

        g.DrawString("Laser Name Keyboard", new Font("Segoe UI", 22, FontStyle.Bold), new SolidBrush(TEXT_PRI), new PointF(box.X + 34, box.Y + 24));
        g.DrawString("Hold the yellow laser on any key for 3 seconds. Press OK when the name is ready.", fontBody, new SolidBrush(TEXT_SEC), new PointF(box.X + 36, box.Y + 62));

        Rectangle nameBox = new Rectangle(box.X + 34, box.Y + 82, box.Width - 68, 52);
        FillRoundRect(g, Color.White, nameBox, 18);
        DrawRoundRect(g, Color.FromArgb(35, BLUE), nameBox, 18);

        string shownName = signupLaserTypedName.Length == 0 ? "Type user name..." : signupLaserTypedName;
        Color shownColor = signupLaserTypedName.Length == 0 ? TEXT_SEC : TEXT_PRI;
        g.DrawString(shownName, new Font("Segoe UI", 18, FontStyle.Bold), new SolidBrush(shownColor), new PointF(nameBox.X + 18, nameBox.Y + 12));

        foreach (KeyValuePair<string, Rectangle> kv in GetSignupKeyboardKeys())
        {
            string key = kv.Key;
            Rectangle rect = kv.Value;
            bool focused = key == signupLaserFocusedKey;
            Color fill = focused ? Color.FromArgb(255, 246, 188) : Color.White;
            Color border = focused ? Color.FromArgb(255, 210, 0) : Color.FromArgb(40, 0, 0, 0);

            if (key == "OK") fill = focused ? Color.FromArgb(205, 245, 230) : GREEN_LIGHT;
            if (key == "CANCEL") fill = focused ? Color.FromArgb(255, 220, 210) : CORAL_LIGHT;

            FillRoundRect(g, fill, rect, 15);
            DrawRoundRect(g, border, rect, 15);

            StringFormat sf = new StringFormat();
            sf.Alignment = StringAlignment.Center;
            sf.LineAlignment = StringAlignment.Center;
            Font keyFont = key.Length == 1 ? new Font("Segoe UI", 17, FontStyle.Bold) : fontH3;
            g.DrawString(key, keyFont, new SolidBrush(TEXT_PRI), rect, sf);

            if (focused && signupLaserFocusStartUtc != DateTime.MinValue)
            {
                double elapsedMs = (DateTime.UtcNow - signupLaserFocusStartUtc).TotalMilliseconds;
                double p = Math.Min(1.0, elapsedMs / SIGNUP_LASER_DWELL_MS);
                Rectangle bar = new Rectangle(rect.X + 8, rect.Bottom - 8, (int)((rect.Width - 16) * p), 4);
                using (SolidBrush b = new SolidBrush(Color.FromArgb(255, 210, 0)))
                    g.FillRectangle(b, bar);

                int remain = Math.Max(0, (int)Math.Ceiling((SIGNUP_LASER_DWELL_MS - elapsedMs) / 1000.0));
                string waitText = remain.ToString() + "s";
                using (SolidBrush wb = new SolidBrush(Color.FromArgb(150, 95, 70, 0)))
                    g.DrawString(waitText, fontTiny, wb, new PointF(rect.Right - 28, rect.Y + 6));
            }
        }
    }

    private bool IsSignupMarker(int markerId)
    {
        return markerId == SIGNUP_NAME_MARKER_ID || markerId == SIGNUP_CHEF_MARKER_ID ||
               markerId == SIGNUP_CLIENT_MARKER_ID || markerId == SIGNUP_CAPTURE_MARKER_ID;
    }

    private string NormalizeRole(string role)
    {
        role = (role ?? "").Trim();
        if (role.Equals("Client", StringComparison.OrdinalIgnoreCase)) return "Client";
        if (role.Equals("Chef", StringComparison.OrdinalIgnoreCase)) return "Chef";
        return "Chef";
    }

    private void EnsureProfileExists(string profileName, string role)
    {
        profileName = (profileName ?? "").Trim();
        role = NormalizeRole(role);
        if (profileName.Length == 0) return;

        LoadProfiles();
        bool exists = userProfiles.Any(p => string.Equals(p.Name, profileName, StringComparison.OrdinalIgnoreCase));
        if (exists) return;

        try
        {
            StringBuilder sb = new StringBuilder();
            if (File.Exists(profileFilePath))
            {
                string old = File.ReadAllText(profileFilePath);
                sb.Append(old);
                if (!old.EndsWith(Environment.NewLine)) sb.AppendLine();
            }

            sb.AppendLine("[" + profileName + "]");
            sb.AppendLine("Name=" + profileName);
            sb.AppendLine("Role=" + role);
            sb.AppendLine("AccentTheme=" + (role.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? "coral" : "blue"));
            sb.AppendLine("WelcomeMessage=Welcome " + profileName + " (" + role + ")!");
            sb.AppendLine("Bio=" + (role.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? "Chef dashboard: recipes, steps, and ingredient control." : "Client dashboard: calories, health score, and meal suggestions."));
            sb.AppendLine("FavoriteMeals=" + (role.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? "Chef Special,Healthy Bowl" : "Healthy Bowl,Light Salad"));
            sb.AppendLine("FavoriteRecipes=" + (role.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? "Tomato Rice,Chicken Rice" : "Light Salad,Tomato Rice"));
            sb.AppendLine("Dislikes=");
            sb.AppendLine("BlockedMarkers=" + (role.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? "" : "8"));
            sb.AppendLine();
            File.WriteAllText(profileFilePath, sb.ToString(), Encoding.UTF8);
            LoadProfiles();
            Console.WriteLine("[PROFILE CREATED] " + profileName + " Role=" + role);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[PROFILE CREATE ERROR] " + ex.Message);
        }
    }

    private bool IsFaceRecognitionMessage(string deviceName, string address)
    {
        return (deviceName ?? "").IndexOf("FaceRecognition", StringComparison.OrdinalIgnoreCase) >= 0 ||
               (address ?? "").IndexOf("Camera", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private void HandleBluetoothSocketMessage(string message)
    {
        try
        {
            Dictionary<string, string> values = ParseBluetoothMessage(message);

            bool detected = false;
            string detectedText;
            if (values.TryGetValue("Detected", out detectedText))
                bool.TryParse(detectedText, out detected);

            string profileName = values.ContainsKey("Profile") ? values["Profile"] : "";
            if (string.IsNullOrWhiteSpace(profileName) && values.ContainsKey("ProfileName"))
                profileName = values["ProfileName"];

            string deviceName = values.ContainsKey("DeviceName") ? values["DeviceName"] : "";
            string address = values.ContainsKey("Address") ? values["Address"] : "";
            string signupStage = values.ContainsKey("Stage") ? values["Stage"] : "";
            string signupMessage = values.ContainsKey("Message") ? values["Message"] : "";
            string signupRole = values.ContainsKey("Role") ? values["Role"] : "";
            string signupName = values.ContainsKey("Name") ? values["Name"] : "";

            if (!string.IsNullOrWhiteSpace(signupStage))
            {
                signupStatusText = string.IsNullOrWhiteSpace(signupMessage) ? ("Signup stage: " + signupStage) : signupMessage;
                if (!string.IsNullOrWhiteSpace(signupName)) pendingSignupName = signupName;
                if (!string.IsNullOrWhiteSpace(signupRole)) pendingSignupRole = signupRole;
                if (signupStage.Equals("CAPTURED", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(signupName))
                {
                    EnsureProfileExists(signupName, signupRole);
                    LoginByFace(signupName);
                    pendingSignupName = "";
                    pendingSignupRole = "";
                }
                Invalidate();
            }

            string emotion = values.ContainsKey("Emotion") ? values["Emotion"] : "Neutral";
            ApplyEmotion(emotion);

            SaveActiveBluetoothFile(detected, profileName, deviceName, address);

            if (!detected || string.IsNullOrWhiteSpace(profileName))
                return;

            lastBluetoothPresenceUtc = DateTime.UtcNow;

            // Face login role rules:
            // - New signup with marker 61/62 keeps the selected role.
            // - Old saved faces that do not have a role are treated as Chef by default.
            if (IsFaceRecognitionMessage(deviceName, address))
                EnsureProfileExists(profileName, string.IsNullOrWhiteSpace(signupRole) ? "Chef" : signupRole);
            else if (!string.IsNullOrWhiteSpace(signupRole))
                EnsureProfileExists(profileName, signupRole);

            ApplyBluetoothMessage(profileName, deviceName, address);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[SOCKET HANDLE ERROR] " + ex.Message);
        }
    }

    private Dictionary<string, string> ParseBluetoothMessage(string message)
    {
        Dictionary<string, string> data = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        string normalized = (message ?? "").Replace(";", "\n");
        string[] parts = normalized.Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string rawPart in parts)
        {
            string part = rawPart.Trim();
            int idx = part.IndexOf('=');
            if (idx <= 0) continue;

            string key = part.Substring(0, idx).Trim();
            string value = part.Substring(idx + 1).Trim();
            data[key] = value;
        }

        return data;
    }

    private DateTime ParseBluetoothTimestampUtc(Dictionary<string, string> values)
    {
        string timestampText = values.ContainsKey("Timestamp") ? values["Timestamp"] : "";
        DateTime parsed;
        if (DateTime.TryParse(timestampText, null, DateTimeStyles.RoundtripKind, out parsed))
            return parsed.ToUniversalTime();

        if (DateTime.TryParse(timestampText, out parsed))
            return parsed.ToUniversalTime();

        return DateTime.MinValue;
    }

    private bool IsBluetoothFileStateFresh(Dictionary<string, string> values)
    {
        DateTime timestampUtc = ParseBluetoothTimestampUtc(values);
        if (timestampUtc == DateTime.MinValue)
            return false;

        TimeSpan age = DateTime.UtcNow - timestampUtc;
        return age.TotalSeconds <= BLUETOOTH_PRESENCE_TIMEOUT_SECONDS;
    }


    private void ReadBluetoothProfile()
    {
        try
        {
            if (!File.Exists(activeBluetoothFilePath)) return;

            string text = File.ReadAllText(activeBluetoothFilePath);
            Dictionary<string, string> values = ParseBluetoothMessage(text);

            bool detected = false;
            string detectedText;
            if (values.TryGetValue("Detected", out detectedText))
                bool.TryParse(detectedText, out detected);

            string profileName = values.ContainsKey("Profile") ? values["Profile"] : "";
            if (string.IsNullOrWhiteSpace(profileName) && values.ContainsKey("ProfileName"))
                profileName = values["ProfileName"];

            string deviceName = values.ContainsKey("DeviceName") ? values["DeviceName"] : "";
            string address = values.ContainsKey("Address") ? values["Address"] : "";

            bool isFresh = IsBluetoothFileStateFresh(values);

            if (detected && !string.IsNullOrWhiteSpace(profileName) && isFresh)
            {
                lastBluetoothPresenceUtc = DateTime.UtcNow;
                ApplyBluetoothMessage(profileName, deviceName, address);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("[BT READ ERROR] " + ex.Message);
        }
    }

    private void SaveActiveBluetoothFile(bool detected, string profileName, string deviceName, string address)
    {
        try
        {
            File.WriteAllText(activeBluetoothFilePath,
                "Detected=" + detected + Environment.NewLine +
                "DeviceName=" + deviceName + Environment.NewLine +
                "Address=" + address + Environment.NewLine +
                "Profile=" + profileName + Environment.NewLine +
                "Timestamp=" + DateTime.UtcNow.ToString("o") + Environment.NewLine);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[BT FILE WRITE ERROR] " + ex.Message);
        }
    }


    private void LogoutToLogin(string reason)
    {
        isLoggedIn = false;
        isBluetoothSessionActive = false;
        activeUserProfile = null;
        blockedMarkers.Clear();
        lastAppliedProfileName = "";
        currentUserName = "Guest";
        currentWelcomeMessage = "Welcome to Smart Kitchen";
        currentBio = "Scan ingredients to get personalized suggestions.";
        currentBluetoothDeviceName = "";
        currentBluetoothAddress = "";
        lastBluetoothPresenceUtc = DateTime.MinValue;
        currentRecipeStepIndex = 0;
        currentRecipeName = "";
        lastGestureCommand = "Waiting for Python gesture connection...";

        lock (objectSync)
        {
            objectList.Clear();
        }

        lock (scannedSync)
        {
            scannedMarkers.Clear();
            scanHistory.Clear();
        }

        ResetMenuSelection(true);
        ApplyAccentTheme("");
        Console.WriteLine("[LOGOUT] " + reason);
        Invalidate();
    }

    private void ApplyBluetoothMessage(string profileName, string deviceName, string address)
    {
        profileName = (profileName ?? "").Trim();
        deviceName = (deviceName ?? "").Trim();
        address = (address ?? "").Trim();

        if (string.IsNullOrWhiteSpace(profileName))
            return;

        lastBluetoothPresenceUtc = DateTime.UtcNow;

        bool sameProfile =
            string.Equals(profileName, lastAppliedProfileName, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(deviceName, currentBluetoothDeviceName, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(address, currentBluetoothAddress, StringComparison.OrdinalIgnoreCase);

        bool needsLogin =
            !isLoggedIn ||
            !isBluetoothSessionActive ||
            activeUserProfile == null ||
            !string.Equals(currentUserName, profileName, StringComparison.OrdinalIgnoreCase);

        if (sameProfile && !needsLogin)
        {
            Invalidate();
            return;
        }

        currentBluetoothDeviceName = deviceName;
        currentBluetoothAddress = address;

        if (this.IsHandleCreated)
        {
            this.BeginInvoke((MethodInvoker)delegate
            {
                Console.WriteLine("[BT APPLY] Re/applying profile: " + profileName);
                ApplyUserProfile(profileName, true);
            });
        }
        else
        {
            Console.WriteLine("[BT APPLY] Re/applying profile: " + profileName);
            ApplyUserProfile(profileName, true);
        }
    }

    private void ApplyUserProfile(string profileName, bool autoLoginByBluetooth)
    {
        LoadProfiles();

        UserProfile profile = userProfiles.FirstOrDefault(p =>
            string.Equals(p.Name, profileName, StringComparison.OrdinalIgnoreCase));

        if (profile == null)
        {
            Console.WriteLine("[PROFILE NOT FOUND] " + profileName + " | File=" + profileFilePath);
            return;
        }

        activeUserProfile = profile;
        currentUserName = profile.Name;
        currentWelcomeMessage = string.IsNullOrWhiteSpace(profile.WelcomeMessage)
            ? ("Welcome " + profile.Name + "!")
            : profile.WelcomeMessage;
        currentBio = string.IsNullOrWhiteSpace(profile.Bio)
            ? "Personalized profile loaded."
            : profile.Bio;

        blockedMarkers = new HashSet<int>(profile.BlockedMarkers);
        lastAppliedProfileName = profile.Name;

        ApplyAccentTheme(profile.AccentTheme);
        RemoveBlockedScans();

        if (autoLoginByBluetooth)
        {
            isBluetoothSessionActive = true;
            isLoggedIn = true;
            ResetMenuSelection(true);
            Console.WriteLine("[BLUETOOTH LOGIN] " + profile.Name + " detected. Opening main page...");
            StartYoloPython();
        }

        Console.WriteLine("[PROFILE APPLIED] " + profile.Name + " | Device=" + currentBluetoothDeviceName + " | Address=" + currentBluetoothAddress + " | BlockedMarkers=" + string.Join(",", blockedMarkers));
        Invalidate();
    }

    private void ApplyAccentTheme(string theme)
    {
        string t = (theme ?? "").Trim().ToLower();

        if (t == "purple")
        {
            currentAccent = PURPLE;
            currentAccentLight = PURPLE_LIGHT;
        }
        else if (t == "blue")
        {
            currentAccent = BLUE;
            currentAccentLight = BLUE_LIGHT;
        }
        else if (t == "coral" || t == "orange")
        {
            currentAccent = CORAL;
            currentAccentLight = CORAL_LIGHT;
        }
        else if (t == "green")
        {
            currentAccent = GREEN;
            currentAccentLight = GREEN_LIGHT;
        }
        else
        {
            currentAccent = BLUE;
            currentAccentLight = BLUE_LIGHT;
        }
    }

    private void RemoveBlockedScans()
    {
        lock (scannedSync)
        {
            foreach (int markerId in blockedMarkers.ToList())
            {
                if (scannedMarkers.ContainsKey(markerId))
                    scannedMarkers.Remove(markerId);
                scanHistory.RemoveAll(x => x == markerId);
            }
        }
    }

    private bool IsMarkerBlocked(int markerId)
    {
        return blockedMarkers.Contains(markerId);
    }

    private List<KitchenItem> GetVisibleItems()
    {
        return items.Where(i => !IsMarkerBlocked(i.MarkerID)).ToList();
    }
    private List<Recipe> GetPersonalizedRecipes()
    {
        List<Recipe> list = recipes
            .Where(r => r.RequiredMarkers.All(mid => !IsMarkerBlocked(mid)))
            .ToList();

        if (contentRecipes != null && contentRecipes.Count > 0)
        {
            foreach (Recipe cr in contentRecipes)
            {
                if (cr.RequiredMarkers == null)
                    cr.RequiredMarkers = GuessMarkersFromContent(cr.Ingredients, cr.Tools);

                if (cr.RequiredMarkers.Any(mid => IsMarkerBlocked(mid)))
                    continue;

                cr.IsContentRecipe = true;
                cr.Emoji = string.IsNullOrWhiteSpace(cr.Emoji) ? "📄" : cr.Emoji;
                cr.BaseKcal = cr.Calories;
                if (cr.Steps == null || cr.Steps.Length == 0)
                    cr.Steps = BuildContentRecipeSteps(cr);

                if (!list.Any(r => string.Equals(r.Name, cr.Name, StringComparison.OrdinalIgnoreCase)))
                    list.Add(cr);
            }
        }

        if (activeUserProfile != null && activeUserProfile.FavoriteMeals.Count > 0)
        {
            list = list
                .OrderByDescending(r => activeUserProfile.FavoriteMeals.Any(f =>
                    string.Equals(f, r.Name, StringComparison.OrdinalIgnoreCase)))
                .ThenByDescending(r => r.IsContentRecipe)
                .ThenBy(r => r.Name)
                .ToList();
        }
        else
        {
            list = list
                .OrderByDescending(r => r.IsContentRecipe)
                .ThenBy(r => r.Name)
                .ToList();
        }

        return list;
    }

    private int[] GuessMarkersFromContent(string ingredients, string tools)
    {
        List<int> markers = new List<int>();
        string text = ((ingredients ?? "") + "," + (tools ?? "")).ToLowerInvariant();

        if (text.Contains("chicken")) markers.Add(0);
        if (text.Contains("tomato")) markers.Add(1);
        if (text.Contains("onion")) markers.Add(2);
        if (text.Contains("spices") || text.Contains("spice")) markers.Add(3);
        if (text.Contains("rice")) markers.Add(4);
        if (text.Contains("oil")) markers.Add(5);
        if (text.Contains("spoon")) markers.Add(6);
        if (text.Contains("pot")) markers.Add(7);
        if (text.Contains("knife")) markers.Add(8);

        return markers.Distinct().OrderBy(x => x).ToArray();
    }

    private Dictionary<int, TuioDemoObject> GetFilteredDetected()
    {
        Dictionary<int, TuioDemoObject> result;
        lock (scannedSync)
        {
            result = new Dictionary<int, TuioDemoObject>(scannedMarkers);
        }

        foreach (int blocked in blockedMarkers.ToList())
        {
            if (result.ContainsKey(blocked))
                result.Remove(blocked);
        }

        return result;
    }

    // =========================
    // Manual utility methods
    // =========================

    private void RemoveLastScannedItem()
    {
        lock (scannedSync)
        {
            if (scanHistory.Count == 0) return;

            int lastMarker = scanHistory[scanHistory.Count - 1];
            scanHistory.RemoveAt(scanHistory.Count - 1);

            if (scannedMarkers.ContainsKey(lastMarker))
                scannedMarkers.Remove(lastMarker);
        }

        Console.WriteLine("[REMOVE] Last scanned marker removed.");
        Invalidate();
    }

    private void StartGestureSocketClient()
    {
        gestureRunning = true;
        gestureThread = new Thread(GestureSocketLoop);
        gestureThread.IsBackground = true;
        gestureThread.Start();
    }

    private void GestureSocketLoop()
    {
        while (gestureRunning)
        {
            try
            {
                if (gestureClient != null)
                {
                    try { gestureClient.Close(); } catch { }
                    gestureClient = null;
                }

                Console.WriteLine("[GESTURE] Connecting to Python on " + GESTURE_SOCKET_HOST + ":" + GESTURE_SOCKET_PORT + " ...");
                gestureClient = new TcpClient();
                gestureClient.Connect(GESTURE_SOCKET_HOST, GESTURE_SOCKET_PORT);
                gestureReader = new StreamReader(gestureClient.GetStream());
                Console.WriteLine("[GESTURE] Connected to Python gesture server.");
                lastGestureCommand = "Python gesture connected.";
                SafeInvalidate();

                while (gestureRunning && gestureClient.Connected)
                {
                    string line = gestureReader.ReadLine();
                    if (line == null) break;

                    string command = line.Trim();
                    if (command.Length == 0) continue;

                    Console.WriteLine("[GESTURE RECEIVED] " + command);
                    HandleGestureCommand(command);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("[GESTURE SOCKET] " + ex.Message);
                lastGestureCommand = "Waiting for Python gesture server...";
                SafeInvalidate();
            }

            if (!gestureRunning) break;
            Thread.Sleep(1500);
        }
    }

    private void HandleGestureCommand(string command)
    {
        if (this.IsDisposed) return;

        if (this.InvokeRequired)
        {
            this.BeginInvoke((MethodInvoker)delegate { HandleGestureCommand(command); });
            return;
        }

        if (command.StartsWith("OBJECT_ROT:", StringComparison.OrdinalIgnoreCase))
        {
            HandleYoloObjectRotation(command);
            return;
        }

        if (command.StartsWith("OBJECT:", StringComparison.OrdinalIgnoreCase))
        {
            HandleYoloObject(command);
            return;
        }

        if (command.StartsWith("LOGIN:", StringComparison.OrdinalIgnoreCase))
        {
            string profileName = command.Substring("LOGIN:".Length).Trim();
            LoginByFace(profileName);
            return;
        }

        if (command.StartsWith("LASER_MENU_INDEX", StringComparison.OrdinalIgnoreCase))
        {
            HandleLaserMenuIndexCommand(command);
            return;
        }

        if (command.StartsWith("LASER_MENU", StringComparison.OrdinalIgnoreCase))
        {
            HandleLaserMenuCommand(command);
            return;
        }

        lastGestureCommand = "Last gesture: " + command;

        if (!isLoggedIn)
        {
            Invalidate();
            return;
        }

        if (command == "NEXT_STEP" || command == "THUMBS_UP" || command == "LIKE")
        {
            GoToNextRecipeStep();
        }
        else if (command == "PREVIOUS_STEP" || command == "THUMBS_DOWN" || command == "DISLIKE")
        {
            GoToPreviousRecipeStep();
        }
        else if (command == "CLEAR_LINES")
        {
            confirmedMenuView = MenuView.Steps;
            currentMenuView = MenuView.Steps;
            menuStatusText = "Gesture lines cleared in Python.";
        }

        Invalidate();
    }

    private Recipe GetReadyRecipe(Dictionary<int, TuioDemoObject> detected)
    {
        HashSet<int> activeMarkers = new HashSet<int>(detected.Keys);

        foreach (Recipe recipe in GetPersonalizedRecipes())
        {
            bool allPresent = true;
            foreach (int id in recipe.RequiredMarkers)
            {
                if (!activeMarkers.Contains(id))
                {
                    allPresent = false;
                    break;
                }
            }

            if (allPresent)
                return recipe;
        }

        return null;
    }

    private void SyncCurrentRecipeStep()
    {
        Dictionary<int, TuioDemoObject> detected;
        lock (scannedSync)
            detected = new Dictionary<int, TuioDemoObject>(scannedMarkers);

        Recipe readyRecipe = GetReadyRecipe(detected);

        if (readyRecipe == null)
        {
            currentRecipeName = "";
            currentRecipeStepIndex = 0;
            return;
        }

        if (!string.Equals(currentRecipeName, readyRecipe.Name, StringComparison.OrdinalIgnoreCase))
        {
            currentRecipeName = readyRecipe.Name;
            currentRecipeStepIndex = 0;
        }

        if (currentRecipeStepIndex < 0) currentRecipeStepIndex = 0;
        if (currentRecipeStepIndex >= readyRecipe.Steps.Length) currentRecipeStepIndex = readyRecipe.Steps.Length - 1;
    }

    private void GoToNextRecipeStep()
    {
        SyncCurrentRecipeStep();

        Dictionary<int, TuioDemoObject> detected;
        lock (scannedSync)
            detected = new Dictionary<int, TuioDemoObject>(scannedMarkers);

        Recipe readyRecipe = GetReadyRecipe(detected);
        confirmedMenuView = MenuView.Steps;
        currentMenuView = MenuView.Steps;

        if (readyRecipe == null)
        {
            menuStatusText = "Complete a recipe first, then use gesture navigation.";
            return;
        }

        if (currentRecipeStepIndex < readyRecipe.Steps.Length - 1)
            currentRecipeStepIndex++;

        menuStatusText = "Gesture NEXT -> " + readyRecipe.Name + " step " + (currentRecipeStepIndex + 1) + "/" + readyRecipe.Steps.Length;
    }

    private void GoToPreviousRecipeStep()
    {
        SyncCurrentRecipeStep();

        Dictionary<int, TuioDemoObject> detected;
        lock (scannedSync)
            detected = new Dictionary<int, TuioDemoObject>(scannedMarkers);

        Recipe readyRecipe = GetReadyRecipe(detected);
        confirmedMenuView = MenuView.Steps;
        currentMenuView = MenuView.Steps;

        if (readyRecipe == null)
        {
            menuStatusText = "Complete a recipe first, then use gesture navigation.";
            return;
        }

        if (currentRecipeStepIndex > 0)
            currentRecipeStepIndex--;

        menuStatusText = "Gesture PREVIOUS -> " + readyRecipe.Name + " step " + (currentRecipeStepIndex + 1) + "/" + readyRecipe.Steps.Length;
    }

    private void SafeInvalidate()
    {
        if (this.IsDisposed) return;

        if (this.IsHandleCreated)
        {
            try
            {
                this.BeginInvoke((MethodInvoker)delegate { Invalidate(); });
            }
            catch { }
        }
    }



    private Dictionary<string, string> ParseSemicolonValues(string command)
    {
        Dictionary<string, string> values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        string[] parts = (command ?? "").Split(';');

        foreach (string part in parts)
        {
            int idx = part.IndexOf('=');
            if (idx > 0)
            {
                string key = part.Substring(0, idx).Trim();
                string value = part.Substring(idx + 1).Trim();
                values[key] = value;
            }
        }

        return values;
    }

    private void HandleLaserMenuIndexCommand(string command)
    {
        if (!isLoggedIn)
        {
            lastGestureCommand = "Laser index detected. Login is required; marker 60 signup keyboard uses LASER_MENU X/Y.";
            Invalidate();
            return;
        }

        Dictionary<string, string> values = ParseSemicolonValues(command);

        int index = -1;
        if (values.ContainsKey("INDEX"))
            int.TryParse(values["INDEX"], out index);

        if (index < 0 || index >= circularMenuItems.Length)
            return;

        currentMenuView = (MenuView)index;
        menuControllerVisible = true;
        lastLaserSeenUtc = DateTime.UtcNow;

        double px, py;
        if (values.ContainsKey("PX") && values.ContainsKey("PY") &&
            double.TryParse(values["PX"], NumberStyles.Any, CultureInfo.InvariantCulture, out px) &&
            double.TryParse(values["PY"], NumberStyles.Any, CultureInfo.InvariantCulture, out py))
        {
            px = Math.Max(0.0, Math.Min(1.0, px));
            py = Math.Max(0.0, Math.Min(1.0, py));

            // Map the camera pointer position onto the circular menu wheel in the C# GUI.
            Rectangle wheelRect = GetCircularMenuWheelRectangle();
            lastLaserNormX = (wheelRect.X + px * wheelRect.Width) / Math.Max(1.0, (double)WIN_W);
            lastLaserNormY = (wheelRect.Y + py * wheelRect.Height) / Math.Max(1.0, (double)WIN_H);
        }
        else
        {
            // Fallback: draw the pointer at the center of the selected slice.
            Rectangle wheelRect = GetCircularMenuWheelRectangle();
            int cx = wheelRect.X + wheelRect.Width / 2;
            int cy = wheelRect.Y + wheelRect.Height / 2;
            double deg = -120.0 + index * (360.0 / circularMenuItems.Length) + (180.0 / circularMenuItems.Length);
            double rad = deg * Math.PI / 180.0;
            double rr = wheelRect.Width * 0.34;
            lastLaserNormX = (cx + Math.Cos(rad) * rr) / Math.Max(1.0, (double)WIN_W);
            lastLaserNormY = (cy + Math.Sin(rad) * rr) / Math.Max(1.0, (double)WIN_H);
        }

        if (index != laserMenuIndex)
        {
            laserMenuIndex = index;
            laserMenuHoldStart = DateTime.Now;
            laserStatusText = "Laser focus: " + circularMenuItems[index] + " • hold yellow laser to select.";
            menuStatusText = laserStatusText;
        }
        else if (laserMenuHoldStart != DateTime.MinValue)
        {
            TimeSpan heldFor = DateTime.Now - laserMenuHoldStart;
            if (heldFor.TotalMilliseconds >= LASER_MENU_HOLD_MS)
            {
                confirmedMenuView = (MenuView)index;
                laserStatusText = "Laser selected: " + circularMenuItems[index];
                menuStatusText = laserStatusText;
                Console.WriteLine("[LASER MENU SELECT] " + circularMenuItems[index]);
            }
        }

        lastGestureCommand = "Laser menu: " + circularMenuItems[index];
        Invalidate();
    }

    private void HandleLaserMenuCommand(string command)
    {
        Dictionary<string, string> values = ParseSemicolonValues(command);

        double nx;
        double ny;

        if (!values.ContainsKey("X") || !values.ContainsKey("Y"))
            return;

        if (!double.TryParse(values["X"], NumberStyles.Any, CultureInfo.InvariantCulture, out nx))
            return;

        if (!double.TryParse(values["Y"], NumberStyles.Any, CultureInfo.InvariantCulture, out ny))
            return;

        nx = Math.Max(0.0, Math.Min(1.0, nx));
        ny = Math.Max(0.0, Math.Min(1.0, ny));

        lastLaserNormX = nx;
        lastLaserNormY = ny;
        lastLaserSeenUtc = DateTime.UtcNow;

        if (signupLaserKeyboardVisible)
        {
            ProcessSignupLaserKeyboard(nx, ny);
            return;
        }

        if (!isLoggedIn)
        {
            lastGestureCommand = "Laser detected. Login is required, or show marker 60 to open signup keyboard.";
            Invalidate();
            return;
        }

        int screenX = (int)(nx * WIN_W);
        int screenY = (int)(ny * WIN_H);

        int centerPanelX = 550;
        int centerPanelY = PAD + 132 + 16;
        Rectangle menuCard = new Rectangle(centerPanelX, centerPanelY + 95, 800, 515);
        Rectangle wheelRect = new Rectangle(menuCard.X + 155, menuCard.Y + 62, 490, 490);

        int cx = wheelRect.X + wheelRect.Width / 2;
        int cy = wheelRect.Y + wheelRect.Height / 2;

        double dx = screenX - cx;
        double dy = screenY - cy;
        double distance = Math.Sqrt(dx * dx + dy * dy);

        int innerRadius = 92;
        int outerRadius = Math.Min(wheelRect.Width, wheelRect.Height) / 2;

        if (distance < innerRadius || distance > outerRadius)
        {
            laserMenuIndex = -1;
            laserMenuHoldStart = DateTime.MinValue;
            laserStatusText = "Laser: point at a menu slice, not outside/center.";
            lastGestureCommand = laserStatusText;
            Invalidate();
            return;
        }

        double angle = Math.Atan2(dy, dx) * 180.0 / Math.PI;
        if (angle < 0.0) angle += 360.0;

        double startBase = -120.0;
        double normalized = angle - startBase;
        while (normalized < 0.0) normalized += 360.0;
        while (normalized >= 360.0) normalized -= 360.0;

        int index = (int)(normalized / (360.0 / circularMenuItems.Length));
        if (index < 0) index = 0;
        if (index >= circularMenuItems.Length) index = circularMenuItems.Length - 1;

        currentMenuView = (MenuView)index;
        menuControllerVisible = true;

        if (index != laserMenuIndex)
        {
            laserMenuIndex = index;
            laserMenuHoldStart = DateTime.Now;
            laserStatusText = "Laser focus: " + circularMenuItems[index] + " • hold yellow laser to select.";
            menuStatusText = laserStatusText;
        }
        else if (laserMenuHoldStart != DateTime.MinValue)
        {
            TimeSpan heldFor = DateTime.Now - laserMenuHoldStart;
            if (heldFor.TotalMilliseconds >= LASER_MENU_HOLD_MS)
            {
                confirmedMenuView = (MenuView)index;
                laserStatusText = "Laser selected: " + circularMenuItems[index];
                menuStatusText = laserStatusText;
                Console.WriteLine("[LASER MENU SELECT] " + circularMenuItems[index]);
            }
        }

        lastGestureCommand = "Laser menu: " + circularMenuItems[index];
        Invalidate();
    }

    private void StopGestureSocketClient()
    {
        try
        {
            gestureRunning = false;
            if (gestureClient != null) gestureClient.Close();
            if (gestureReader != null) gestureReader.Close();
            if (gestureThread != null && gestureThread.IsAlive) gestureThread.Join(500);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[GESTURE STOP ERROR] " + ex.Message);
        }
    }

    private void StopBluetoothListener()
    {
        try
        {
            bluetoothServerRunning = false;

            if (bluetoothTimer != null)
                bluetoothTimer.Stop();

            if (bluetoothServer != null)
                bluetoothServer.Stop();

            if (bluetoothServerThread != null && bluetoothServerThread.IsAlive)
                bluetoothServerThread.Join(500);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[SOCKET STOP ERROR] " + ex.Message);
        }
    }

    // =========================
    // Face Identification + YOLO Object Tracking
    // =========================
    private void StartFaceLoginWatcher()
    {
        try
        {
            faceLoginTimer = new System.Windows.Forms.Timer();
            faceLoginTimer.Interval = 700;
            faceLoginTimer.Tick += (s, e) => { ReadFaceLoginFile(); };
            faceLoginTimer.Start();
        }
        catch (Exception ex)
        {
            Console.WriteLine("[FACE TIMER ERROR] " + ex.Message);
        }
    }

    private void ReadFaceLoginFile()
    {
        try
        {
            if (!File.Exists(activeFaceFilePath)) return;

            string text = File.ReadAllText(activeFaceFilePath).Trim();
            if (string.IsNullOrWhiteSpace(text)) return;

            string profileName = "";

            if (text.StartsWith("LOGIN:", StringComparison.OrdinalIgnoreCase))
            {
                profileName = text.Substring("LOGIN:".Length).Trim();
            }
            else
            {
                Dictionary<string, string> values = ParseBluetoothMessage(text);
                bool detected = false;
                string detectedText;
                if (values.TryGetValue("Detected", out detectedText))
                    bool.TryParse(detectedText, out detected);

                if (!detected && !values.ContainsKey("Detected")) detected = true;

                if (detected)
                {
                    if (values.ContainsKey("Profile")) profileName = values["Profile"];
                    else if (values.ContainsKey("ProfileName")) profileName = values["ProfileName"];
                    else if (values.ContainsKey("Name")) profileName = values["Name"];
                    else if (values.ContainsKey("User")) profileName = values["User"];
                }
            }

            if (!string.IsNullOrWhiteSpace(profileName))
                LoginByFace(profileName);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[FACE LOGIN READ ERROR] " + ex.Message);
        }
    }

    private void LoginByFace(string profileName)
    {
        profileName = (profileName ?? "").Trim();
        if (profileName.Length == 0) return;

        if (isLoggedIn && string.Equals(currentUserName, profileName, StringComparison.OrdinalIgnoreCase))
            return;

        isLoggedIn = true;
        isBluetoothSessionActive = false;

        lock (objectSync) { objectList.Clear(); }
        lock (scannedSync)
        {
            scannedMarkers.Clear();
            scanHistory.Clear();
        }

        ResetMenuSelection(true);

        LoadProfiles();
        UserProfile profile = userProfiles.FirstOrDefault(p =>
            string.Equals(p.Name, profileName, StringComparison.OrdinalIgnoreCase));

        if (profile != null)
        {
            ApplyUserProfile(profile.Name, false);
        }
        else
        {
            activeUserProfile = null;
            blockedMarkers.Clear();
            currentUserName = profileName;
            currentWelcomeMessage = "Welcome " + profileName + "!";
            currentBio = "Face identification login completed.";
            ApplyAccentTheme("");
        }

        Console.WriteLine("[FACE LOGIN SUCCESS] " + profileName + " detected. Opening main page...");
        StartYoloPython();
        Invalidate();
    }

    private void StartFaceRecognition()
    {
        try
        {
            if (faceRecognitionProcess != null && !faceRecognitionProcess.HasExited)
                return;

            string projectDir = AppDomain.CurrentDomain.BaseDirectory;
            string pythonFile = Path.Combine(projectDir, "smart_kitchen_all_in_one_camera.py");
            if (!File.Exists(pythonFile))
                pythonFile = Path.Combine(projectDir, "face_identification.py");

            if (!File.Exists(pythonFile))
            {
                Console.WriteLine("[FACE] smart_kitchen_all_in_one_camera.py / face_identification.py not found: " + pythonFile);
                return;
            }

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = "py";
            startInfo.Arguments = "\"" + pythonFile + "\"";
            startInfo.WorkingDirectory = projectDir;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = false;
            startInfo.WindowStyle = ProcessWindowStyle.Normal;

            faceRecognitionProcess = Process.Start(startInfo);
            Console.WriteLine("[FACE] Python all-in-one / face recognition started from: " + pythonFile);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[FACE START ERROR] " + ex.Message);
        }
    }


    private void StopFaceRecognition()
    {
        try
        {
            if (faceRecognitionProcess != null && !faceRecognitionProcess.HasExited)
            {
                faceRecognitionProcess.Kill();
                faceRecognitionProcess = null;
                Console.WriteLine("[FACE] Face Identification stopped.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("[FACE STOP ERROR] " + ex.Message);
        }
    }

    private void StartYoloPython()
    {
        try
        {
            // smart_kitchen_all_in_one_camera.py already contains YOLO + Gaze + Face Recognition.
            // Do not open a second Python camera process if the all-in-one process is running.
            if (faceRecognitionProcess != null && !faceRecognitionProcess.HasExited)
                return;

            if (yoloProcess != null && !yoloProcess.HasExited)
                return;

            string projectDir = AppDomain.CurrentDomain.BaseDirectory;
            string pythonFile = Path.Combine(projectDir, "smart_kitchen_all_in_one_camera.py");
            if (!File.Exists(pythonFile))
                pythonFile = Path.Combine(projectDir, "yolo_tracking.py");

            if (!File.Exists(pythonFile))
            {
                Console.WriteLine("[YOLO] smart_kitchen_all_in_one_camera.py / yolo_tracking.py not found: " + pythonFile);
                return;
            }

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = "py";
            startInfo.Arguments = "\"" + pythonFile + "\"";
            startInfo.WorkingDirectory = projectDir;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = false;
            startInfo.WindowStyle = ProcessWindowStyle.Normal;

            yoloProcess = Process.Start(startInfo);
            Console.WriteLine("[YOLO] Python tracking started from: " + pythonFile);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[YOLO START ERROR] " + ex.Message);
        }
    }


    private void StopYoloPython()
    {
        try
        {
            if (yoloProcess != null && !yoloProcess.HasExited)
            {
                yoloProcess.Kill();
                yoloProcess = null;
                Console.WriteLine("[YOLO] YOLO stopped.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("[YOLO STOP ERROR] " + ex.Message);
        }
    }



    private void HandleYoloObjectRotation(string command)
    {
        if (!isLoggedIn)
        {
            lastGestureCommand = "YOLO rotation detected, but login is required.";
            Invalidate();
            return;
        }

        // Expected format from Python:
        // OBJECT_ROT:Tomato;ANGLE=1.5708;DEG=90.0
        string payload = command.Substring("OBJECT_ROT:".Length).Trim();
        string[] parts = payload.Split(';');
        if (parts.Length == 0) return;

        string obj = parts[0].Trim();
        int markerId = GetMarkerIdFromYoloName(obj);
        if (markerId == -1) return;
        if (IsMarkerBlocked(markerId)) return;

        float angle = 0f;
        for (int i = 1; i < parts.Length; i++)
        {
            string part = parts[i].Trim();
            int idx = part.IndexOf('=');
            if (idx <= 0) continue;

            string key = part.Substring(0, idx).Trim();
            string value = part.Substring(idx + 1).Trim();

            if (key.Equals("ANGLE", StringComparison.OrdinalIgnoreCase))
            {
                float.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out angle);
            }
        }

        while (angle < 0f) angle += (float)(Math.PI * 2.0);
        while (angle >= (float)(Math.PI * 2.0)) angle -= (float)(Math.PI * 2.0);

        lock (scannedSync)
        {
            scannedMarkers[markerId] = null; // null means YOLO source, angle comes from yoloMarkerAngles
            yoloMarkerAngles[markerId] = angle;

            if (scanHistory.Contains(markerId))
                scanHistory.Remove(markerId);

            scanHistory.Add(markerId);
        }

        int deg = (int)(angle * 180f / (float)Math.PI);
        lastGestureCommand = "YOLO detected: " + obj + " | virtual rotation " + deg + "°";
        Console.WriteLine("[YOLO ROTATION] " + obj + " -> Marker " + markerId + " angle=" + deg + "°");
        Invalidate();
    }

    private float GetMarkerAngle(int markerId, TuioDemoObject obj)
    {
        if (obj != null) return obj.Angle;

        lock (scannedSync)
        {
            if (yoloMarkerAngles.ContainsKey(markerId))
                return yoloMarkerAngles[markerId];
        }

        return 0f;
    }

    private void HandleYoloObject(string command)
    {
        if (!isLoggedIn)
        {
            lastGestureCommand = "YOLO object detected, but login is required.";
            Invalidate();
            return;
        }

        string obj = command.Replace("OBJECT:", "").Trim();
        int markerId = GetMarkerIdFromYoloName(obj);

        if (markerId == -1) return;
        if (IsMarkerBlocked(markerId)) return;

        lock (scannedSync)
        {
            if (!scannedMarkers.ContainsKey(markerId))
            {
                scannedMarkers[markerId] = null;
                scanHistory.Add(markerId);
            }
        }

        lastGestureCommand = "YOLO detected: " + obj;
        Console.WriteLine("[YOLO DETECTED] " + obj + " -> Marker " + markerId);
        Invalidate();
    }

    private int GetMarkerIdFromYoloName(string obj)
    {
        string o = (obj ?? "").Trim().ToLowerInvariant();

        if (o == "spoon") return 6;
        if (o == "pot" || o == "bowl" || o == "cup") return 7;
        if (o == "knife") return 8;
        if (o == "tomato" || o == "apple" || o == "orange" || o == "carrot" || o == "broccoli") return 1;
        if (o == "chicken" || o == "sandwich") return 0;
        if (o == "rice" || o == "pizza") return 4;
        if (o == "oil" || o == "bottle") return 5;
        if (o == "onion") return 2;
        if (o == "spices" || o == "spice") return 3;

        return -1;
    }


    // =========================
    // Facial Emotion Adaptive UI
    // =========================
    private void ApplyEmotion(string emotion)
    {
        currentEmotion = (emotion ?? "Neutral").Trim();

        emotionHappyMode = currentEmotion.Equals("Happy", StringComparison.OrdinalIgnoreCase);
        emotionSadMode = currentEmotion.Equals("Sad", StringComparison.OrdinalIgnoreCase);
        emotionAngryMode = currentEmotion.Equals("Angry", StringComparison.OrdinalIgnoreCase);

        if (emotionHappyMode)
        {
            adaptiveBackground = Color.FromArgb(235, 250, 240);
        }
        else if (emotionSadMode)
        {
            adaptiveBackground = Color.FromArgb(225, 235, 250);
        }
        else if (emotionAngryMode)
        {
            adaptiveBackground = Color.FromArgb(255, 228, 228);
        }
        else if (currentEmotion.Equals("Surprised", StringComparison.OrdinalIgnoreCase))
        {
            adaptiveBackground = Color.FromArgb(255, 248, 220);
        }
        else
        {
            adaptiveBackground = Color.FromArgb(250, 248, 243);
        }

        this.BackColor = adaptiveBackground;
        SafeInvalidate();
    }

    private string GetEmotionText()
    {
        if (currentEmotion.Equals("Happy", StringComparison.OrdinalIgnoreCase)) return "Happy 🙂";
        if (currentEmotion.Equals("Sad", StringComparison.OrdinalIgnoreCase)) return "Sad 😔";
        if (currentEmotion.Equals("Angry", StringComparison.OrdinalIgnoreCase)) return "Angry 😠";
        if (currentEmotion.Equals("Surprised", StringComparison.OrdinalIgnoreCase)) return "Surprised 😮";
        return "Neutral 😐";
    }

    private void Form_Closing(object sender, CancelEventArgs e)
    {
        // Save gaze/user report before closing the app
        SaveGazeUserReport();

        try { if (faceLoginTimer != null) faceLoginTimer.Stop(); } catch { }
        StopYoloPython();
        StopFaceRecognition();
        StopBluetoothListener();
        StopGestureSocketClient();
        StopGazeSocketClient();
        client.removeTuioListener(this);
        client.disconnect();
        Environment.Exit(0);
    }

    //gaze_control

    private string GetDominantGazeZone()
    {
        if (gazeLeftHits >= gazeCenterHits && gazeLeftHits >= gazeRightHits)
            return "Recipes View";
        if (gazeCenterHits >= gazeLeftHits && gazeCenterHits >= gazeRightHits)
            return "Overview View";
        return "Calories View";
    }

    private void SaveGazeUserReport()
    {
        try
        {
            int total = gazeLeftHits + gazeCenterHits + gazeRightHits;
            double durationSeconds = Math.Round((DateTime.UtcNow - gazeSessionStartUtc).TotalSeconds, 2);
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "smart_kitchen_gaze_user_report.csv");

            using (StreamWriter sw = new StreamWriter(path, false, Encoding.UTF8))
            {
                sw.WriteLine("Smart Kitchen Gaze User Report");
                sw.WriteLine("Generated At," + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
                sw.WriteLine("User Name," + currentUserName);
                sw.WriteLine("Login Status," + (isLoggedIn ? "Logged In" : "Guest/Not Logged In"));
                sw.WriteLine("Device," + (string.IsNullOrWhiteSpace(currentBluetoothDeviceName) ? "Manual login" : currentBluetoothDeviceName));
                sw.WriteLine("Session Duration Seconds," + durationSeconds.ToString(CultureInfo.InvariantCulture));
                sw.WriteLine("");
                sw.WriteLine("View,Hits,Percentage");
                sw.WriteLine("Recipes View," + gazeLeftHits + "," + PercentText(gazeLeftHits, total));
                sw.WriteLine("Overview View," + gazeCenterHits + "," + PercentText(gazeCenterHits, total));
                sw.WriteLine("Calories View," + gazeRightHits + "," + PercentText(gazeRightHits, total));
                sw.WriteLine("");
                sw.WriteLine("Total Valid Gaze Hits," + total);
                sw.WriteLine("Most Looked View," + (total == 0 ? "No gaze hits yet" : GetDominantGazeZone()));
                sw.WriteLine("Adaptive Rule,LEFT=Recipes CENTER=Overview RIGHT=Calories");
            }

            Console.WriteLine("[GAZE REPORT SAVED] " + path);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[GAZE REPORT ERROR] " + ex.Message);
        }
    }

    private string PercentText(int value, int total)
    {
        if (total <= 0) return "0%";
        return Math.Round((value * 100.0) / total, 2).ToString(CultureInfo.InvariantCulture) + "%";
    }

    private void StopGazeSocketClient()
    {
        try
        {
            gazeRunning = false;
            if (gazeClient != null) gazeClient.Close();
            if (gazeReader != null) gazeReader.Close();
            if (gazeThread != null && gazeThread.IsAlive) gazeThread.Join(500);
        }
        catch (Exception ex)
        {
            Console.WriteLine("[GAZE STOP ERROR] " + ex.Message);
        }
    }
    //
    // =========================
    // TUIO
    // =========================
    public void addTuioObject(TuioObject o)
    {
        TuioDemoObject demoObj = new TuioDemoObject(o);
        ////
        if (o.SymbolID == UNITY_AR_MARKER_ID)
        {
            StartUnityAR();
            return;
        }
        ////
        if (IsSignupMarker(o.SymbolID))
        {
            TriggerSignupMarkerFromTuio(o.SymbolID);
            return;
        }

        if (!isLoggedIn)
        {
            if (o.SymbolID == LOGIN_MARKER_ID)
            {
                isLoggedIn = true;
                isBluetoothSessionActive = false;

                lock (objectSync)
                {
                    objectList.Clear();
                }

                lock (scannedSync)
                {
                    scannedMarkers.Clear();
                    scanHistory.Clear();
                }

                ResetMenuSelection(true);
                Console.WriteLine("[LOGIN SUCCESS] Marker 25 detected. Opening main page...");
                StartYoloPython();
                Invalidate();
            }
            else
            {
                Console.WriteLine("[LOGIN BLOCKED] Waiting for marker 25 or Bluetooth user. Detected Marker: " + o.SymbolID);
            }

            return;
        }

        lock (objectSync)
        {
            objectList[o.SessionID] = demoObj;
        }

        if (o.SymbolID == HEATMAP_MARKER_ID)
        {
            ToggleHeatmapByMarker();
            return;
        }

        if (o.SymbolID == MENU_MARKER_ID)
        {
            menuControllerVisible = true;
            UpdateCircularMenuState(o.Angle);
            Console.WriteLine("[MENU] Circular menu controller detected.");
            return;
        }

        if (IsCrudMarker(o.SymbolID))
        {
            HandleCrudMarker(o.SymbolID);
            return;
        }

        if (IsMarkerBlocked(o.SymbolID))
        {
            Console.WriteLine("[BLOCKED] Marker " + o.SymbolID + " ignored for current user profile.");
            return;
        }

        lock (scannedSync)
        {
            scannedMarkers[o.SymbolID] = demoObj;

            if (scanHistory.Contains(o.SymbolID))
                scanHistory.Remove(o.SymbolID);

            scanHistory.Add(o.SymbolID);
        }

        Console.WriteLine("[SCAN] Marker " + o.SymbolID + " saved.");
    }

    public void updateTuioObject(TuioObject o)
    {
        if (IsSignupMarker(o.SymbolID))
        {
            TriggerSignupMarkerFromTuio(o.SymbolID);
            return;
        }

        if (!isLoggedIn)
        {
            if (o.SymbolID == LOGIN_MARKER_ID)
            {
                isLoggedIn = true;
                isBluetoothSessionActive = false;

                lock (objectSync)
                {
                    objectList.Clear();
                }

                lock (scannedSync)
                {
                    scannedMarkers.Clear();
                    scanHistory.Clear();
                }

                ResetMenuSelection(true);
                Console.WriteLine("[LOGIN SUCCESS] Marker 25 detected during update. Opening main page...");
                StartYoloPython();
                Invalidate();
            }

            return;
        }

        TuioDemoObject demoObj = new TuioDemoObject(o);

        lock (objectSync)
        {
            objectList[o.SessionID] = demoObj;
        }

        if (o.SymbolID == HEATMAP_MARKER_ID)
        {
            // Marker 50 works as a toggle only when it first appears.
            // Keeping it in view or updating it must not keep forcing the heatmap on.
            return;
        }

        if (o.SymbolID == MENU_MARKER_ID)
        {
            menuControllerVisible = true;
            UpdateCircularMenuState(o.Angle);
            return;
        }

        if (IsCrudMarker(o.SymbolID))
            return;

        if (IsMarkerBlocked(o.SymbolID))
            return;

        lock (scannedSync)
        {
            scannedMarkers[o.SymbolID] = demoObj;
        }
    }

    public void removeTuioObject(TuioObject o)
    {
        if (!isLoggedIn)
            return;

        lock (objectSync)
        {
            if (objectList.ContainsKey(o.SessionID))
                objectList.Remove(o.SessionID);
        }

        if (o.SymbolID == MENU_MARKER_ID)
        {
            menuControllerVisible = false;
            lastMenuIndex = -1;
            menuHoldStart = DateTime.MinValue;
            menuStatusText = "Menu controller removed. Show marker 20 again to rotate and choose.";
            Console.WriteLine("[MENU] Circular menu controller removed from camera view.");
            return;
        }

        if (o.SymbolID == HEATMAP_MARKER_ID)
        {
            // Do not hide the heatmap when marker 50 leaves the camera.
            // Show marker 50 again to toggle it off.
            return;
        }

        Console.WriteLine("[LIVE REMOVED] Marker " + o.SymbolID + " removed from camera view but kept in scanned list.");
    }

    public void addTuioCursor(TuioCursor c) { }
    public void updateTuioCursor(TuioCursor c) { }
    public void removeTuioCursor(TuioCursor c) { }
    public void addTuioBlob(TuioBlob b) { }
    public void updateTuioBlob(TuioBlob b) { }
    public void removeTuioBlob(TuioBlob b) { }

    public void refresh(TuioTime t)
    {
        Invalidate();
    }

    // =========================
    // Menu
    // =========================
    private void ResetMenuSelection(bool keepOverview)
    {
        currentMenuView = MenuView.Overview;
        confirmedMenuView = keepOverview ? MenuView.Overview : confirmedMenuView;
        lastMenuIndex = -1;
        menuHoldStart = DateTime.MinValue;
        menuControllerVisible = false;
        menuStatusText = " ";
    }

    private float NormalizeDegrees(float angle)
    {
        float deg = angle * 180f / (float)Math.PI;
        while (deg < 0f) deg += 360f;
        while (deg >= 360f) deg -= 360f;
        return deg;
    }

    private int GetMenuIndexFromAngle(float angle)
    {
        float deg = NormalizeDegrees(angle);
        float slice = 360f / circularMenuItems.Length;
        int index = (int)(deg / slice);
        if (index < 0) index = 0;
        if (index >= circularMenuItems.Length) index = circularMenuItems.Length - 1;
        return index;
    }

    private void UpdateCircularMenuState(float angle)
    {
        int index = GetMenuIndexFromAngle(angle);
        currentMenuView = (MenuView)index;

        if (index != lastMenuIndex)
        {
            lastMenuIndex = index;
            menuHoldStart = DateTime.Now;
            menuStatusText = "Current focus: " + circularMenuItems[index] + "  •  Hold the marker steady for 1 second to select.";
        }
        else if (menuHoldStart != DateTime.MinValue)
        {
            TimeSpan heldFor = DateTime.Now - menuHoldStart;
            if (heldFor.TotalMilliseconds >= MENU_HOLD_MS && confirmedMenuView != (MenuView)index)
            {
                confirmedMenuView = (MenuView)index;
                menuStatusText = "Selected: " + circularMenuItems[index] + "  •  Rotate again to choose another section.";
                Console.WriteLine("[MENU SELECT] " + circularMenuItems[index]);
            }
        }
    }

    private TuioDemoObject GetLiveMenuControllerObject()
    {
        lock (objectSync)
        {
            foreach (TuioDemoObject obj in objectList.Values)
            {
                if (obj.SymbolID == MENU_MARKER_ID)
                    return obj;
            }
        }

        return null;
    }

    // =========================
    // Paint
    // =========================
    protected override void OnPaintBackground(PaintEventArgs e)
    {
        Graphics g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;

        g.Clear(adaptiveBackground);

        if (!isLoggedIn)
        {
            DrawLoginScreen(g);
            DrawSignupLaserKeyboard(g);
            DrawLaserPointerOverlay(g);
            return;
        }

        Dictionary<int, TuioDemoObject> detected = GetFilteredDetected();

        DrawHeader(g);
        DrawGazeAdaptiveLayout(g, detected);
    }



    private Rectangle GetCircularMenuWheelRectangle()
    {
        int centerPanelX = 550;
        int centerPanelY = PAD + 132 + 16;
        Rectangle menuCard = new Rectangle(centerPanelX, centerPanelY + 95, 800, 515);
        return new Rectangle(menuCard.X + 155, menuCard.Y + 62, 490, 490);
    }

    private void DrawLaserPointerOverlay(Graphics g)
    {
        if (lastLaserSeenUtc == DateTime.MinValue) return;
        if ((DateTime.UtcNow - lastLaserSeenUtc).TotalMilliseconds > 1200) return;
        if (lastLaserNormX < 0 || lastLaserNormY < 0) return;

        int x = (int)(lastLaserNormX * WIN_W);
        int y = (int)(lastLaserNormY * WIN_H);

        Color laserYellow = Color.FromArgb(255, 230, 0);
        using (Pen laserPen = new Pen(laserYellow, 4))
        using (SolidBrush laserBrush = new SolidBrush(Color.FromArgb(95, laserYellow)))
        using (SolidBrush textBrush = new SolidBrush(Color.FromArgb(120, 95, 0)))
        {
            g.FillEllipse(laserBrush, x - 20, y - 20, 40, 40);
            g.DrawEllipse(laserPen, x - 20, y - 20, 40, 40);
            g.DrawLine(laserPen, x - 32, y, x + 32, y);
            g.DrawLine(laserPen, x, y - 32, x, y + 32);

            if (!string.IsNullOrWhiteSpace(laserStatusText))
                g.DrawString(laserStatusText, fontSmall, textBrush, new PointF(Math.Max(16, x + 26), Math.Max(160, y - 10)));
        }
    }

    private void DrawGazeAdaptiveLayout(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        GraphicsState state;

        if (IsClientMode())
        {
            state = g.Save();
            if (IsGazeZoomActive(MenuView.Recipes))
            {
                g.TranslateTransform(-8, -5);
                g.ScaleTransform(1.04f, 1.04f);
            }
            DrawClientMealCatalog(g, detected);
            g.Restore(state);

            state = g.Save();
            if (IsGazeZoomActive(MenuView.Overview))
            {
                g.TranslateTransform(-10, -6);
                g.ScaleTransform(1.04f, 1.04f);
            }
            DrawClientCenterPanel(g, detected);
            g.Restore(state);

            state = g.Save();
            if (IsGazeZoomActive(MenuView.Calories))
            {
                g.TranslateTransform(-18, -6);
                g.ScaleTransform(1.04f, 1.04f);
            }
            DrawClientRightPanel(g, detected);
            g.Restore(state);

            DrawFooterMappingBar(g);
            DrawGazeFocusBadge(g);
            DrawFreeGazeMagnifier(g, detected);
            DrawLaserPointerOverlay(g);
            return;
        }

        state = g.Save();
        if (IsGazeZoomActive(MenuView.Recipes))
        {
            g.TranslateTransform(-8, -5);
            g.ScaleTransform(1.04f, 1.04f);
        }
        DrawIngredientGrid(g, detected);
        g.Restore(state);

        state = g.Save();
        if (IsGazeZoomActive(MenuView.Overview))
        {
            g.TranslateTransform(-10, -6);
            g.ScaleTransform(1.04f, 1.04f);
        }
        DrawCenterPanel(g, detected);
        g.Restore(state);

        state = g.Save();
        if (IsGazeZoomActive(MenuView.Calories))
        {
            g.TranslateTransform(-18, -6);
            g.ScaleTransform(1.04f, 1.04f);
        }
        DrawRightPanel(g, detected);
        g.Restore(state);

        DrawFooterMappingBar(g);
        DrawGazeFocusBadge(g);
        DrawFreeGazeMagnifier(g, detected);
        DrawLaserPointerOverlay(g);

        if (heatmapVisible)
            DrawEyeGazeHeatmap(g);
        else
            DrawHeatmapHint(g);
    }


    private void AddHeatmapPointFromGaze(string mainCommand, Dictionary<string, string> values)
    {
        // Free gaze mode: Python sends ScreenX/ScreenY in normalized GUI coordinates.
        // Marker 50 only reveals the heatmap; the heatmap itself is built from continuous gaze points.
        float x = WIN_W * 0.50f;
        float y = WIN_H * 0.50f;
        bool hasFreePoint = false;

        if (values.ContainsKey("ScreenX") && values.ContainsKey("ScreenY"))
        {
            double sx, sy;
            if (double.TryParse(values["ScreenX"], NumberStyles.Any, CultureInfo.InvariantCulture, out sx) &&
                double.TryParse(values["ScreenY"], NumberStyles.Any, CultureInfo.InvariantCulture, out sy))
            {
                sx = Math.Max(0.0, Math.Min(1.0, sx));
                sy = Math.Max(0.0, Math.Min(1.0, sy));
                x = (float)(sx * WIN_W);
                y = (float)(sy * WIN_H);
                hasFreePoint = true;
            }
        }

        // Backward compatibility for older Python code that only sent LEFT / CENTER / RIGHT.
        if (!hasFreePoint)
        {
            string direction = "";
            if (mainCommand == "GAZE_LEFT") direction = "LEFT";
            else if (mainCommand == "GAZE_CENTER") direction = "CENTER";
            else if (mainCommand == "GAZE_RIGHT") direction = "RIGHT";
            else if (mainCommand == "GAZE_SAMPLE" && values.ContainsKey("Direction"))
                direction = values["Direction"].Trim().ToUpperInvariant();

            if (direction != "LEFT" && direction != "CENTER" && direction != "RIGHT") return;

            if (direction == "LEFT") x = WIN_W * 0.23f;
            else if (direction == "CENTER") x = WIN_W * 0.50f;
            else x = WIN_W * 0.77f;

            y = WIN_H * 0.52f;
            x += heatmapRandom.Next(-95, 96);
            y += heatmapRandom.Next(-75, 76);
        }
        else
        {
            // Tiny jitter makes repeated fixation visible as a warm cluster instead of one dot.
            x += heatmapRandom.Next(-8, 9);
            y += heatmapRandom.Next(-8, 9);
        }

        x = Math.Max(20, Math.Min(WIN_W - 20, x));
        y = Math.Max(135, Math.Min(WIN_H - 45, y));

        double freeSxForZoom = x / (double)WIN_W;
        double freeSyForZoom = y / (double)WIN_H;
        if (values.ContainsKey("ScreenX"))
            double.TryParse(values["ScreenX"], NumberStyles.Any, CultureInfo.InvariantCulture, out freeSxForZoom);
        if (values.ContainsKey("ScreenY"))
            double.TryParse(values["ScreenY"], NumberStyles.Any, CultureInfo.InvariantCulture, out freeSyForZoom);

        if (values.ContainsKey("Direction"))
        {
            string d = values["Direction"].Trim().Replace("_", " ");
            lastGazeDirection = "Looking at screen point: " + d + "  X=" + ((int)x).ToString() + " Y=" + ((int)y).ToString();
        }

        lock (heatmapSync)
        {
            heatmapPoints.Add(new PointF(x, y));
            if (heatmapPoints.Count > MAX_HEATMAP_POINTS)
                heatmapPoints.RemoveRange(0, heatmapPoints.Count - MAX_HEATMAP_POINTS);
        }
    }


    private void DrawFreeGazeMagnifier(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        if (!IsFreeGazePointLive()) return;

        float cx = lastFreeGazePoint.X;
        float cy = lastFreeGazePoint.Y;
        int radius = 92;

        // Do not cover the top header too much.
        if (cy < 135) cy = 135;
        if (cy > WIN_H - 70) cy = WIN_H - 70;
        if (cx < 110) cx = 110;
        if (cx > WIN_W - 110) cx = WIN_W - 110;

        using (GraphicsPath clip = new GraphicsPath())
        {
            clip.AddEllipse(cx - radius, cy - radius, radius * 2, radius * 2);
            GraphicsState st = g.Save();
            g.SetClip(clip);

            // Draw the same GUI area again, scaled around the exact gaze point.
            g.TranslateTransform(cx, cy);
            g.ScaleTransform(1.0f, 1.0f);
            g.TranslateTransform(-cx, -cy);

            if (cx < WIN_W * 0.34f)
                DrawIngredientGrid(g, detected);
            else if (cx > WIN_W * 0.66f)
                DrawRightPanel(g, detected);
            else
                DrawCenterPanel(g, detected);

            g.Restore(st);
        }


    }

    private void DrawHeatmapHint(Graphics g)
    {
        Rectangle hint = new Rectangle(WIN_W - 455, WIN_H - 86, 420, 46);
        FillRoundRect(g, Color.FromArgb(245, 245, 245), hint, 18);
        DrawRoundRect(g, Color.FromArgb(35, 0, 0, 0), hint, 18);
        g.DrawString("Eye Heatmap hidden  •  Show marker 50 once to show / again to hide", fontSmall, new SolidBrush(TEXT_SEC), new PointF(hint.X + 18, hint.Y + 15));
    }

    private void DrawEyeGazeHeatmap(Graphics g)
    {
        List<PointF> pts;
        lock (heatmapSync)
        {
            pts = new List<PointF>(heatmapPoints);
        }

        Rectangle area = new Rectangle(44, 165, WIN_W - 88, WIN_H - 220);

        // Transparent heatmap layer over the real GUI, not a black board.
        using (SolidBrush softLayer = new SolidBrush(Color.FromArgb(32, 255, 255, 255)))
            g.FillRectangle(softLayer, area);

        FillRoundRect(g, Color.FromArgb(232, 255, 255, 255), new Rectangle(area.X, area.Y, area.Width, 56), 20);
        g.DrawString("Eye Gaze Heatmap  •  Marker 50", fontH2, new SolidBrush(TEXT_PRI), new PointF(area.X + 22, area.Y + 14));
        g.DrawString("User looks at GUI; Python sends gaze samples; marker 50 reveals this heatmap.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(area.X + 300, area.Y + 20));

        Rectangle leftZone = new Rectangle(area.X, area.Y + 56, area.Width / 3, area.Height - 56);
        Rectangle centerZone = new Rectangle(area.X + area.Width / 3, area.Y + 56, area.Width / 3, area.Height - 56);
        Rectangle rightZone = new Rectangle(area.X + (area.Width / 3) * 2, area.Y + 56, area.Width - (area.Width / 3) * 2, area.Height - 56);

        using (SolidBrush b1 = new SolidBrush(Color.FromArgb(20, 52, 120, 246))) g.FillRectangle(b1, leftZone);
        using (SolidBrush b2 = new SolidBrush(Color.FromArgb(20, 29, 158, 117))) g.FillRectangle(b2, centerZone);
        using (SolidBrush b3 = new SolidBrush(Color.FromArgb(20, 216, 90, 48))) g.FillRectangle(b3, rightZone);

        using (Pen p = new Pen(Color.FromArgb(135, 255, 255, 255), 2))
        {
            g.DrawLine(p, centerZone.X, area.Y + 56, centerZone.X, area.Bottom);
            g.DrawLine(p, rightZone.X, area.Y + 56, rightZone.X, area.Bottom);
        }

        g.DrawString("Ingredients", fontH3, Brushes.White, new PointF(leftZone.X + 22, leftZone.Y + 16));
        g.DrawString("Dashboard", fontH3, Brushes.White, new PointF(centerZone.X + 22, centerZone.Y + 16));
        g.DrawString("Nutrition", fontH3, Brushes.White, new PointF(rightZone.X + 22, rightZone.Y + 16));

        foreach (PointF pt in pts)
        {
            DrawHeatSpot(g, pt.X, pt.Y, 68, Color.FromArgb(34, 60, 90, 255));
            DrawHeatSpot(g, pt.X, pt.Y, 42, Color.FromArgb(54, 0, 210, 90));
            DrawHeatSpot(g, pt.X, pt.Y, 25, Color.FromArgb(72, 255, 230, 0));
            DrawHeatSpot(g, pt.X, pt.Y, 13, Color.FromArgb(112, 255, 55, 0));
        }

        FillRoundRect(g, Color.FromArgb(235, 255, 255, 255), new Rectangle(area.X + 18, area.Bottom - 54, 390, 36), 16);
        g.DrawString("Heat points: " + pts.Count + "  •  " + heatmapStatus, fontSmall, new SolidBrush(TEXT_PRI), new PointF(area.X + 34, area.Bottom - 44));
    }

    private void DrawHeatSpot(Graphics g, float x, float y, int radius, Color color)
    {
        using (GraphicsPath path = new GraphicsPath())
        {
            path.AddEllipse(x - radius, y - radius, radius * 2, radius * 2);
            using (PathGradientBrush brush = new PathGradientBrush(path))
            {
                brush.CenterColor = color;
                brush.SurroundColors = new Color[] { Color.FromArgb(0, color.R, color.G, color.B) };
                g.FillPath(brush, path);
            }
        }
    }

    private void DrawGazeFocusBadge(Graphics g)
    {
        if ((DateTime.UtcNow - lastGazeZoomUtc).TotalMilliseconds >= 1700) return;

        string title = gazeZoomView == MenuView.Recipes ? "Left Focus" :
                       gazeZoomView == MenuView.Calories ? "Right Focus" : "Center Focus";

        Rectangle badge = new Rectangle(WIN_W - 360, 112, 320, 40);
        FillRoundRect(g, Color.FromArgb(236, 255, 248), badge, 18);
        DrawRoundRect(g, Color.FromArgb(29, 158, 117), badge, 18);
        using (SolidBrush b = new SolidBrush(TEAL_DARK))
            g.DrawString(title + "  •  Zoom", fontH3, b, new PointF(badge.X + 16, badge.Y + 10));
    }

    private void DrawLoginScreen(Graphics g)
    {
        Rectangle hero = new Rectangle(58, 58, WIN_W - 116, WIN_H - 116);
        Rectangle leftPanel = new Rectangle(hero.X + 26, hero.Y + 26, 445, hero.Height - 52);
        Rectangle rightPanel = new Rectangle(leftPanel.Right + 26, hero.Y + 26, hero.Width - leftPanel.Width - 78, hero.Height - 52);

        using (LinearGradientBrush bgBrush = new LinearGradientBrush(
            new Rectangle(0, 0, WIN_W, WIN_H),
            Color.FromArgb(255, 252, 247),
            Color.FromArgb(238, 247, 243),
            45f))
        {
            g.FillRectangle(bgBrush, 0, 0, WIN_W, WIN_H);
        }

        using (SolidBrush soft = new SolidBrush(Color.FromArgb(24, currentAccent)))
        {
            g.FillEllipse(soft, WIN_W - 410, -120, 520, 520);
            g.FillEllipse(soft, -160, WIN_H - 360, 440, 440);
        }

        FillRoundRect(g, Color.FromArgb(252, 255, 253), hero, 34);
        DrawRoundRect(g, Color.FromArgb(38, 0, 0, 0), hero, 34);

        FillRoundRect(g, Color.FromArgb(255, 247, 238), leftPanel, 30);
        DrawRoundRect(g, Color.FromArgb(28, 0, 0, 0), leftPanel, 30);

        FillRoundRect(g, CARD_BG, rightPanel, 30);
        DrawRoundRect(g, Color.FromArgb(26, 0, 0, 0), rightPanel, 30);

        Rectangle logoBox = new Rectangle(leftPanel.X + 26, leftPanel.Y + 26, 54, 54);
        FillRoundRect(g, Color.FromArgb(255, 255, 255), logoBox, 16);
        DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), logoBox, 16);
        DrawEmoji(g, "👨‍🍳", new Rectangle(logoBox.X + 8, logoBox.Y + 7, 42, 42));

        g.DrawString("Smart Kitchen", fontTitle, new SolidBrush(TEXT_PRI), new PointF(leftPanel.X + 94, leftPanel.Y + 25));
        g.DrawString("Tangible cooking access system", fontSmall, new SolidBrush(CORAL), new PointF(leftPanel.X + 96, leftPanel.Y + 57));

        g.DrawString("Start cooking\nwith marker, face\nor Bluetooth", new Font("Segoe UI", 25, FontStyle.Bold), new SolidBrush(TEAL_DARK), new RectangleF(leftPanel.X + 28, leftPanel.Y + 112, leftPanel.Width - 56, 132));
        g.DrawString("A professional smart-kitchen dashboard that opens by marker 25, known face recognition, or a saved Bluetooth profile.", fontBody, new SolidBrush(TEXT_SEC), new RectangleF(leftPanel.X + 30, leftPanel.Y + 254, leftPanel.Width - 60, 70));

        Rectangle statusStrip = new Rectangle(leftPanel.X + 30, leftPanel.Y + 340, leftPanel.Width - 60, 54);
        FillRoundRect(g, Color.FromArgb(236, 255, 248), statusStrip, 18);
        DrawRoundRect(g, Color.FromArgb(35, GREEN), statusStrip, 18);
        DrawEmoji(g, "●", new Rectangle(statusStrip.X + 14, statusStrip.Y + 14, 24, 24));
        g.DrawString("System ready", fontH3, new SolidBrush(TEAL_DARK), new PointF(statusStrip.X + 46, statusStrip.Y + 10));
        g.DrawString("reacTIVision + Python services can connect automatically.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(statusStrip.X + 46, statusStrip.Y + 31));

        Rectangle feature1 = new Rectangle(leftPanel.X + 30, leftPanel.Y + 418, leftPanel.Width - 60, 68);
        Rectangle feature2 = new Rectangle(leftPanel.X + 30, leftPanel.Y + 500, leftPanel.Width - 60, 68);
        Rectangle feature3 = new Rectangle(leftPanel.X + 30, leftPanel.Y + 582, leftPanel.Width - 60, 68);

        FillRoundRect(g, GREEN_LIGHT, feature1, 20);
        FillRoundRect(g, BLUE_LIGHT, feature2, 20);
        FillRoundRect(g, AMBER_LIGHT, feature3, 20);

        DrawEmoji(g, "📶", new Rectangle(feature1.X + 16, feature1.Y + 15, 32, 32));
        g.DrawString("Bluetooth profile login", fontH3, new SolidBrush(TEXT_PRI), new PointF(feature1.X + 58, feature1.Y + 11));
        g.DrawString("Loads name, role, theme, and preferred meals.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(feature1.X + 58, feature1.Y + 35));

        DrawEmoji(g, "🙂", new Rectangle(feature2.X + 16, feature2.Y + 15, 32, 32));
        g.DrawString("Face recognition signup", fontH3, new SolidBrush(TEXT_PRI), new PointF(feature2.X + 58, feature2.Y + 11));
        g.DrawString("Marker 60→name, 61/62→role, 63→capture.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(feature2.X + 58, feature2.Y + 35));

        DrawEmoji(g, "🧭", new Rectangle(feature3.X + 16, feature3.Y + 15, 32, 32));
        g.DrawString("TUIO marker control", fontH3, new SolidBrush(TEXT_PRI), new PointF(feature3.X + 58, feature3.Y + 11));
        g.DrawString("Marker 20 menu, marker 50 heatmap toggle.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(feature3.X + 58, feature3.Y + 35));

        Rectangle topBadge = new Rectangle(rightPanel.X + 30, rightPanel.Y + 28, 178, 36);
        FillRoundRect(g, currentAccentLight, topBadge, 18);
        g.DrawString("ACCESS PANEL", fontSmall, new SolidBrush(currentAccent), new PointF(topBadge.X + 22, topBadge.Y + 10));

        g.DrawString("Waiting for login", new Font("Segoe UI", 24, FontStyle.Bold), new SolidBrush(TEXT_PRI), new PointF(rightPanel.X + 30, rightPanel.Y + 82));
        g.DrawString("Choose one access method below. The original login logic is unchanged.", fontBody, new SolidBrush(TEXT_SEC), new PointF(rightPanel.X + 32, rightPanel.Y + 124));

        Rectangle methods = new Rectangle(rightPanel.X + 34, rightPanel.Y + 178, rightPanel.Width - 68, 190);
        FillRoundRect(g, Color.FromArgb(246, 245, 238), methods, 24);
        DrawRoundRect(g, Color.FromArgb(18, 0, 0, 0), methods, 24);
        g.DrawString("Login flow", fontH2, new SolidBrush(TEXT_PRI), new PointF(methods.X + 24, methods.Y + 18));

        int rowY = methods.Y + 58;
        string[] rows = new string[] {
            "1   Open reacTIVision and show marker 25 for manual login.",
            "2   Known face or Bluetooth profile opens the dashboard automatically.",
            "3   Unknown face: show marker 60 and enter the new user name.",
            "4   Select role: marker 61 = Chef, marker 62 = Client, marker 63 = capture."
        };
        for (int i = 0; i < rows.Length; i++)
        {
            Rectangle bullet = new Rectangle(methods.X + 24, rowY + i * 30 + 2, 20, 20);
            FillRoundRect(g, i == 0 ? GREEN_LIGHT : BLUE_LIGHT, bullet, 10);
            g.DrawString(rows[i], fontBody, new SolidBrush(TEXT_SEC), new PointF(methods.X + 56, rowY + i * 30));
        }

        Rectangle signupBox = new Rectangle(rightPanel.X + 34, rightPanel.Y + 394, rightPanel.Width - 68, 112);
        FillRoundRect(g, Color.FromArgb(255, 242, 232), signupBox, 22);
        DrawRoundRect(g, Color.FromArgb(35, CORAL), signupBox, 22);
        DrawEmoji(g, "🪪", new Rectangle(signupBox.X + 18, signupBox.Y + 18, 34, 34));
        g.DrawString("Face signup status", fontH3, new SolidBrush(CORAL), new PointF(signupBox.X + 62, signupBox.Y + 17));
        g.DrawString(signupStatusText, fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(signupBox.X + 62, signupBox.Y + 43, signupBox.Width - 82, 56));

        Rectangle quick = new Rectangle(rightPanel.X + 34, signupBox.Bottom + 24, rightPanel.Width - 68, 94);
        FillRoundRect(g, Color.FromArgb(245, 250, 255), quick, 22);
        DrawRoundRect(g, Color.FromArgb(28, BLUE), quick, 22);
        g.DrawString("Quick markers", fontH3, new SolidBrush(BLUE), new PointF(quick.X + 18, quick.Y + 14));
        g.DrawString("M25 Login   •   M60 Name   •   M61 Chef   •   M62 Client   •   M63 Capture", fontBody, new SolidBrush(TEXT_PRI), new PointF(quick.X + 18, quick.Y + 43));
        g.DrawString("After login: M20 Menu   •   M50 Heatmap   •   M11–M14 CRUD Studio", fontSmall, new SolidBrush(TEXT_SEC), new PointF(quick.X + 18, quick.Y + 68));

        Rectangle footer = new Rectangle(rightPanel.X + 34, rightPanel.Bottom - 110, rightPanel.Width - 68, 76);
        FillRoundRect(g, currentAccentLight, footer, 20);
        DrawRoundRect(g, Color.FromArgb(24, currentAccent), footer, 20);
        DrawEmoji(g, "📡", new Rectangle(footer.X + 16, footer.Y + 18, 34, 34));
        g.DrawString("Current Bluetooth status", fontH3, new SolidBrush(currentAccent), new PointF(footer.X + 60, footer.Y + 13));
        g.DrawString(
            currentBluetoothDeviceName == "" ? "No active Bluetooth user applied yet." :
            ("Device: " + currentBluetoothDeviceName + "  •  Profile: " + currentUserName),
            fontSmall,
            new SolidBrush(TEXT_PRI),
            new RectangleF(footer.X + 60, footer.Y + 40, footer.Width - 80, 28)
        );
    }

    private void DrawEmoji(Graphics g, string emoji, Rectangle rect)
    {
        TextRenderer.DrawText(
            g,
            emoji,
            fontEmoji,
            rect,
            Color.Black,
            TextFormatFlags.Left |
            TextFormatFlags.Top |
            TextFormatFlags.NoPadding |
            TextFormatFlags.NoPrefix
        );
    }

    private void DrawHeader(Graphics g)
    {
        Rectangle r = new Rectangle(PAD, PAD, WIN_W - PAD * 2, 132);
        FillRoundRect(g, CARD_BG, r, 14);
        DrawRoundRect(g, Color.FromArgb(28, 0, 0, 0), r, 14);

        DrawEmoji(g, "🍽", new Rectangle(r.X + 20, r.Y + 18, 36, 36));
        g.DrawString("Smart Kitchen", fontTitle, new SolidBrush(TEXT_PRI), new PointF(r.X + 68, r.Y + 14));
        string modeLine = IsClientMode()
            ? "Client Ready Meals: Pizza, Pasta, Burger, Wrap, Salad, Smoothie  •  M20 = Menu  •  M25 = Login  •  M50 = Heatmap"
            : "Physical Objects Enabled: Spoon, Pot, Knife  •  M11–M14 = CRUD Studio  •  M20 = Menu  •  M25 = Login  •  M50 = Heatmap";
        g.DrawString(
            modeLine,
            fontSmall, new SolidBrush(TEXT_SEC), new PointF(r.X + 22, r.Y + 58)
        );
        string roleText = activeUserProfile == null ? "Guest" : (string.IsNullOrWhiteSpace(activeUserProfile.Role) ? "Client" : activeUserProfile.Role);
        g.DrawString(
            "User: " + currentUserName +
            "  •  Role: " + roleText +
            "  •  Mood: " + GetEmotionText() +
            "  •  Device: " +
            (string.IsNullOrWhiteSpace(currentBluetoothDeviceName) ? "Face/Manual login" : currentBluetoothDeviceName),
            fontSmall, new SolidBrush(currentAccent), new PointF(r.X + 22, r.Y + 84)
        );

        if (emotionAngryMode)
        {
            Rectangle helpRect = new Rectangle(r.Right - 690, r.Y + 82, 130, 28);
            FillRoundRect(g, Color.FromArgb(220, 70, 70), helpRect, 12);
            g.DrawString("HELP MODE", fontH3, new SolidBrush(Color.White), new PointF(helpRect.X + 20, helpRect.Y + 5));
        }

        if (emotionAngryMode)
        {
            g.DrawString("Need help? Interface is simplifying suggestions.", fontSmall, Brushes.Red, new PointF(r.X + 535, r.Y + 105));
        }
        else if (emotionSadMode)
        {
            g.DrawString("Calm mode: quick and simple suggestions.", fontSmall, new SolidBrush(BLUE), new PointF(r.X + 535, r.Y + 105));
        }
        else if (emotionHappyMode)
        {
            g.DrawString("Positive mode: richer recommendations enabled.", fontSmall, new SolidBrush(GREEN), new PointF(r.X + 535, r.Y + 105));
        }

        Rectangle msgRect = new Rectangle(r.Right - 505, r.Y + 22, 485, 86);
        Color roleLight = roleText.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? CORAL_LIGHT : BLUE_LIGHT;
        Color roleLight2 = roleText.Equals("Chef", StringComparison.OrdinalIgnoreCase) ? AMBER_LIGHT : Color.FromArgb(224, 236, 255);
        using (LinearGradientBrush br = new LinearGradientBrush(msgRect, roleLight, roleLight2, 0f))
            g.FillPath(br, RoundedRect(msgRect, 12));
        string roleDashboardText = roleText.Equals("Chef", StringComparison.OrdinalIgnoreCase) ?
            "Chef Mode: recipes, ingredients, steps, and tools." :
            "Client Mode: calories, health score, and suggestions.";
        g.DrawString(currentWelcomeMessage + "  •  " + roleDashboardText, fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(msgRect.X + 16, msgRect.Y + 24, msgRect.Width - 32, msgRect.Height - 18));

        g.DrawString(
            "Gaze: " + lastGazeDirection +
            " | L:" + gazeLeftHits +
            " C:" + gazeCenterHits +
            " R:" + gazeRightHits,
            fontSmall,
            new SolidBrush(currentAccent),
            new PointF(r.X + 22, r.Y + 105)
        );
    }

    private bool IsClientMode()
    {
        return activeUserProfile != null &&
               !string.IsNullOrWhiteSpace(activeUserProfile.Role) &&
               activeUserProfile.Role.Equals("Client", StringComparison.OrdinalIgnoreCase);
    }

    private int GetClientTotalCalories(Dictionary<int, TuioDemoObject> detected)
    {
        int total = 0;
        foreach (ClientMeal meal in clientMeals)
        {
            if (detected.ContainsKey(meal.MarkerID))
                total += meal.Calories;
        }
        return total;
    }

    private int GetClientAverageHealth(Dictionary<int, TuioDemoObject> detected)
    {
        List<ClientMeal> active = clientMeals.Where(m => detected.ContainsKey(m.MarkerID)).ToList();
        if (active.Count == 0) return 0;
        return (int)Math.Round(active.Average(m => m.HealthScore));
    }

    private List<ClientMeal> GetActiveClientMeals(Dictionary<int, TuioDemoObject> detected)
    {
        return clientMeals.Where(m => detected.ContainsKey(m.MarkerID)).OrderBy(m => m.MarkerID).ToList();
    }

    private ClientMeal GetBestClientMeal(Dictionary<int, TuioDemoObject> detected)
    {
        List<ClientMeal> active = GetActiveClientMeals(detected);
        if (active.Count > 0)
            return active.OrderByDescending(m => m.HealthScore).ThenBy(m => m.Calories).First();
        return clientMeals.OrderByDescending(m => m.HealthScore).ThenBy(m => m.Calories).First();
    }

    private void DrawClientDashboard(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        DrawClientMealCatalog(g, detected);
        DrawClientCenterPanel(g, detected);
        DrawClientRightPanel(g, detected);
    }

    private void DrawClientMealCatalog(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        int startY = PAD + 132 + 34;
        int panelX = 56;
        int panelW = 470;
        int cellW = (panelW - 14) / 2;
        int cellH = 118;

        g.DrawString("Ready Meals", fontH2, new SolidBrush(TEXT_PRI), new PointF(panelX, startY));
        g.DrawString("Client menu: scan markers to select prepared meals.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(panelX, startY + 24));
        startY += 54;

        for (int i = 0; i < clientMeals.Length; i++)
        {
            ClientMeal meal = clientMeals[i];
            int col = i % 2;
            int row = i / 2;
            int x = panelX + col * (cellW + 14);
            int y = startY + row * (cellH + 10);
            bool active = detected.ContainsKey(meal.MarkerID);
            DrawClientMealCard(g, meal, new Rectangle(x, y, cellW, cellH), active);
        }

        int rows = (int)Math.Ceiling(clientMeals.Length / 2.0);
        int y2 = startY + rows * (cellH + 10) + 12;
        DrawClientQuickPreferencePanel(g, new Rectangle(panelX, y2, panelW, 130));
    }

    private void DrawClientMealCard(Graphics g, ClientMeal meal, Rectangle r, bool active)
    {
        FillRoundRect(g, active ? meal.LightColor : CARD_BG, r, 12);
        using (Pen pen = new Pen(active ? meal.AccentColor : Color.FromArgb(30, 0, 0, 0), active ? 2 : 1))
            DrawRoundRectPen(g, pen, r, 12);

        Rectangle badge = new Rectangle(r.Right - 34, r.Y + 7, 28, 18);
        FillRoundRect(g, active ? meal.AccentColor : GRAY_LIGHT, badge, 9);
        g.DrawString("M" + meal.MarkerID, fontMarker, new SolidBrush(active ? Color.White : GRAY_MID), new PointF(badge.X + 4, badge.Y + 2));

        DrawEmoji(g, meal.Emoji, new Rectangle(r.X + 8, r.Y + 8, 30, 30));
        g.DrawString(meal.Name, fontH3, new SolidBrush(TEXT_PRI), new RectangleF(r.X + 44, r.Y + 9, r.Width - 82, 38));
        g.DrawString(meal.Category, fontTiny, new SolidBrush(TEXT_SEC), new PointF(r.X + 44, r.Y + 47));
        g.DrawString(meal.Calories + " kcal", fontH3, new SolidBrush(active ? meal.AccentColor : TEXT_PRI), new PointF(r.X + 10, r.Y + 72));
        g.DrawString("Health " + meal.HealthScore + "/100", fontSmall, new SolidBrush(active ? meal.AccentColor : TEXT_SEC), new PointF(r.X + 104, r.Y + 77));

        if (active)
        {
            Rectangle selected = new Rectangle(r.Right - 80, r.Bottom - 28, 68, 20);
            FillRoundRect(g, meal.AccentColor, selected, 10);
            g.DrawString("Selected", fontTiny, new SolidBrush(Color.White), new PointF(selected.X + 10, selected.Y + 4));
        }
    }

    private void DrawClientQuickPreferencePanel(Graphics g, Rectangle r)
    {
        FillRoundRect(g, Color.FromArgb(233, 242, 255), r, CARD_R);
        DrawRoundRect(g, Color.FromArgb(45, 52, 120, 246), r, CARD_R);
        g.DrawString("Client Preferences", fontH3, new SolidBrush(BLUE), new PointF(r.X + 14, r.Y + 10));
        string name = currentUserName == "Guest" ? "Client" : currentUserName;
        g.DrawString("Profile: " + name + "  •  Mode: ready meals and nutrition", fontSmall, new SolidBrush(TEXT_PRI), new PointF(r.X + 14, r.Y + 38));
        string dislikes = activeUserProfile != null && !string.IsNullOrWhiteSpace(activeUserProfile.Dislikes) ? activeUserProfile.Dislikes : "No avoid-list saved";
        g.DrawString("Avoid: " + dislikes, fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 14, r.Y + 62, r.Width - 28, 20));
        g.DrawString("Tip: scan one or more ready-meal markers to compare calories and health score.", fontTiny, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 14, r.Y + 92, r.Width - 28, 28));
    }

    private void DrawClientCenterPanel(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        int x = 560;
        int y = PAD + 132 + 24;
        int w = 760;

        int total = GetClientTotalCalories(detected);
        int avgHealth = GetClientAverageHealth(detected);
        List<ClientMeal> active = GetActiveClientMeals(detected);
        ClientMeal best = GetBestClientMeal(detected);

        Rectangle summary = new Rectangle(x, y, w, 110);
        using (LinearGradientBrush br = new LinearGradientBrush(summary, Color.FromArgb(224, 236, 255), Color.FromArgb(232, 247, 243), 0f))
            g.FillPath(br, RoundedRect(summary, 16));
        g.DrawString("Client Nutrition Dashboard", fontH2, new SolidBrush(TEXT_PRI), new PointF(summary.X + 20, summary.Y + 16));
        g.DrawString(active.Count + " selected ready meals", fontSmall, new SolidBrush(TEXT_SEC), new PointF(summary.X + 20, summary.Y + 46));
        g.DrawString(total.ToString(), fontBig, new SolidBrush(BLUE), new PointF(summary.Right - 260, summary.Y + 16));
        g.DrawString("total kcal", fontSmall, new SolidBrush(TEXT_SEC), new PointF(summary.Right - 250, summary.Y + 64));
        g.DrawString(avgHealth == 0 ? "--" : avgHealth + "/100", fontBig, new SolidBrush(avgHealth >= 75 ? GREEN : (avgHealth >= 55 ? AMBER : CORAL)), new PointF(summary.Right - 120, summary.Y + 16));
        g.DrawString("health", fontSmall, new SolidBrush(TEXT_SEC), new PointF(summary.Right - 112, summary.Y + 64));

        Rectangle hero = new Rectangle(x, y + 135, w, 280);
        FillRoundRect(g, CARD_BG, hero, 18);
        DrawRoundRect(g, Color.FromArgb(34, 0, 0, 0), hero, 18);
        DrawEmoji(g, best.Emoji, new Rectangle(hero.X + 26, hero.Y + 24, 72, 72));
        g.DrawString(active.Count == 0 ? "Recommended Meal" : "Best Selected Meal", fontSmall, new SolidBrush(BLUE), new PointF(hero.X + 118, hero.Y + 28));
        g.DrawString(best.Name, new Font("Segoe UI", 24, FontStyle.Bold), new SolidBrush(TEXT_PRI), new RectangleF(hero.X + 116, hero.Y + 52, hero.Width - 160, 42));
        g.DrawString(best.Description, fontBody, new SolidBrush(TEXT_SEC), new RectangleF(hero.X + 118, hero.Y + 98, hero.Width - 150, 54));

        DrawClientMetricPill(g, new Rectangle(hero.X + 24, hero.Y + 174, 160, 70), "Calories", best.Calories + " kcal", best.AccentColor, best.LightColor);
        DrawClientMetricPill(g, new Rectangle(hero.X + 204, hero.Y + 174, 130, 70), "Protein", best.Protein + "g", GREEN, GREEN_LIGHT);
        DrawClientMetricPill(g, new Rectangle(hero.X + 354, hero.Y + 174, 130, 70), "Carbs", best.Carbs + "g", AMBER, AMBER_LIGHT);
        DrawClientMetricPill(g, new Rectangle(hero.X + 504, hero.Y + 174, 130, 70), "Fat", best.Fat + "g", CORAL, CORAL_LIGHT);

        Rectangle tagArea = new Rectangle(hero.X + 24, hero.Bottom - 36, hero.Width - 48, 22);
        int tx = tagArea.X;
        foreach (string tag in best.Tags)
        {
            int tw = (int)g.MeasureString(tag, fontTiny).Width + 22;
            Rectangle tr = new Rectangle(tx, tagArea.Y, tw, 20);
            FillRoundRect(g, best.LightColor, tr, 10);
            g.DrawString(tag, fontTiny, new SolidBrush(best.AccentColor), new PointF(tr.X + 10, tr.Y + 4));
            tx += tw + 8;
        }

        Rectangle list = new Rectangle(x, y + 440, w, 350);
        FillRoundRect(g, Color.FromArgb(250, 248, 243), list, 18);
        DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), list, 18);
        g.DrawString("Selected Meal Breakdown", fontH2, new SolidBrush(TEXT_PRI), new PointF(list.X + 22, list.Y + 18));

        if (active.Count == 0)
        {
            g.DrawString("No client meal markers selected yet. Scan Pizza, Pasta, Burger, Salad, or Dessert markers to build an order.", fontBody, new SolidBrush(TEXT_SEC), new RectangleF(list.X + 22, list.Y + 62, list.Width - 44, 58));
            DrawEmoji(g, "🧾", new Rectangle(list.Right - 120, list.Y + 95, 86, 86));
        }
        else
        {
            int rowY = list.Y + 62;
            foreach (ClientMeal meal in active.Take(5))
            {
                Rectangle row = new Rectangle(list.X + 18, rowY, list.Width - 36, 48);
                FillRoundRect(g, meal.LightColor, row, 12);
                DrawEmoji(g, meal.Emoji, new Rectangle(row.X + 10, row.Y + 8, 28, 28));
                g.DrawString(meal.Name, fontH3, new SolidBrush(TEXT_PRI), new PointF(row.X + 48, row.Y + 8));
                g.DrawString(meal.Calories + " kcal  •  " + meal.Protein + "g protein  •  " + meal.PrepMinutes + " min", fontSmall, new SolidBrush(TEXT_SEC), new PointF(row.X + 48, row.Y + 29));
                g.DrawString(meal.HealthScore + "/100", fontH3, new SolidBrush(meal.HealthScore >= 75 ? GREEN : (meal.HealthScore >= 55 ? AMBER : CORAL)), new PointF(row.Right - 82, row.Y + 14));
                rowY += 58;
            }
        }
    }

    private void DrawClientMetricPill(Graphics g, Rectangle r, string title, string value, Color accent, Color light)
    {
        FillRoundRect(g, light, r, 14);
        g.DrawString(title, fontTiny, new SolidBrush(TEXT_SEC), new PointF(r.X + 12, r.Y + 10));
        g.DrawString(value, fontH3, new SolidBrush(accent), new PointF(r.X + 12, r.Y + 34));
    }

    private void DrawClientRightPanel(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        int x = 1370;
        int startY = PAD + 132 + 34;
        int w = WIN_W - x - 58;
        if (w < 430)
        {
            x = WIN_W - 500;
            w = 450;
        }

        List<ClientMeal> active = GetActiveClientMeals(detected);
        ClientMeal best = GetBestClientMeal(detected);

        Rectangle profile = new Rectangle(x, startY, w, 190);
        FillRoundRect(g, Color.FromArgb(229, 244, 255), profile, CARD_R);
        using (Pen pen = new Pen(BLUE, 2)) DrawRoundRectPen(g, pen, profile, CARD_R);
        g.DrawString("👤 Client Profile", fontH3, new SolidBrush(TEXT_PRI), new PointF(profile.X + 18, profile.Y + 18));
        g.DrawString("Name: " + currentUserName, fontBody, new SolidBrush(TEXT_PRI), new PointF(profile.X + 18, profile.Y + 52));
        string ageText = activeUserProfile != null && activeUserProfile.Age > 0 ? activeUserProfile.Age.ToString() : "Not set";
        g.DrawString("Age: " + ageText + "  •  Goal: smarter ready-meal choices", fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(profile.X + 18, profile.Y + 78, profile.Width - 36, 22));
        string fav = activeUserProfile != null && activeUserProfile.FavoriteMeals.Count > 0 ? string.Join(", ", activeUserProfile.FavoriteMeals.ToArray()) : "Pizza, Pasta, Healthy Wraps";
        g.DrawString("Favorites: " + fav, fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(profile.X + 18, profile.Y + 104, profile.Width - 36, 38));
        string avoid = activeUserProfile != null && !string.IsNullOrWhiteSpace(activeUserProfile.Dislikes) ? activeUserProfile.Dislikes : "extra oil, too much sugar";
        g.DrawString("Avoid: " + avoid, fontSmall, new SolidBrush(CORAL), new RectangleF(profile.X + 18, profile.Y + 148, profile.Width - 36, 22));

        startY += 212;
        Rectangle suggestion = new Rectangle(x, startY, w, 245);
        FillRoundRect(g, CARD_BG, suggestion, CARD_R);
        DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), suggestion, CARD_R);
        g.DrawString("Smart Recommendation", fontH2, new SolidBrush(TEXT_PRI), new PointF(suggestion.X + 18, suggestion.Y + 20));
        DrawEmoji(g, best.Emoji, new Rectangle(suggestion.Right - 112, suggestion.Y + 48, 82, 82));
        g.DrawString(best.Name, fontH3, new SolidBrush(best.AccentColor), new RectangleF(suggestion.X + 18, suggestion.Y + 58, suggestion.Width - 140, 28));
        string why = best.HealthScore >= 80 ? "Great choice for a client: balanced calories and strong nutrition score." :
                     best.HealthScore >= 60 ? "Good choice, but keep portion size moderate." :
                     "Treat meal: pair it with water or salad to improve the order.";
        g.DrawString(why, fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(suggestion.X + 18, suggestion.Y + 92, suggestion.Width - 150, 60));
        g.DrawString("Prep time: " + best.PrepMinutes + " min", fontBody, new SolidBrush(TEXT_PRI), new PointF(suggestion.X + 18, suggestion.Y + 162));
        g.DrawString("Health score: " + best.HealthScore + "/100", fontBody, new SolidBrush(best.HealthScore >= 75 ? GREEN : (best.HealthScore >= 55 ? AMBER : CORAL)), new PointF(suggestion.X + 18, suggestion.Y + 190));

        startY += 270;
        Rectangle notes = new Rectangle(x, startY, w, 210);
        FillRoundRect(g, Color.FromArgb(255, 249, 239), notes, CARD_R);
        DrawRoundRect(g, Color.FromArgb(45, 224, 174, 82), notes, CARD_R);
        g.DrawString("Client Notes", fontH2, new SolidBrush(TEXT_PRI), new PointF(notes.X + 18, notes.Y + 18));
        string line1 = active.Count == 0 ? "Scan a ready meal marker to start a client order." : "Order contains " + active.Count + " ready meal item(s).";
        g.DrawString("• " + line1, fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(notes.X + 18, notes.Y + 60, notes.Width - 36, 24));
        g.DrawString("• Use M20 to keep the circular menu available for navigation.", fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(notes.X + 18, notes.Y + 88, notes.Width - 36, 24));
        g.DrawString("• Client mode hides chef tools and focuses on calories, meals, and preferences.", fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(notes.X + 18, notes.Y + 116, notes.Width - 36, 48));
    }

    private void DrawIngredientGrid(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        int startY = PAD + 132 + 34;
        int panelX = 56;
        int panelW = 420;
        int cellW = (panelW - 14) / 2;
        int cellH = 112;

        g.DrawString("Scanned Items", fontH2, new SolidBrush(TEXT_PRI), new PointF(panelX, startY));
        startY += 34;

        List<KitchenItem> visibleItems = GetVisibleItems();

        for (int i = 0; i < visibleItems.Count; i++)
        {
            KitchenItem item = visibleItems[i];
            int col = i % 2;
            int row = i / 2;
            int x = panelX + col * (cellW + 14);
            int y = startY + row * (cellH + 10);

            bool isActive = detected.ContainsKey(item.MarkerID);
            DrawItemCard(g, item, new Rectangle(x, y, cellW, cellH), isActive, isActive ? detected[item.MarkerID] : null);
        }

        int rows = (int)Math.Ceiling(visibleItems.Count / 2.0);
        int legendY = startY + rows * (cellH + 10) + 8;
        DrawRotationLegend(g, new Rectangle(panelX, legendY, panelW, 82));

        legendY += 94;
        if (!emotionSadMode && !emotionAngryMode)
        {
            DrawPhysicalObjectPanel(g, detected, new Rectangle(panelX, legendY, panelW, 122));
        }
    }

    private void DrawItemCard(Graphics g, KitchenItem item, Rectangle r, bool active, TuioDemoObject obj)
    {
        Color bg = active ? item.LightColor : CARD_BG;
        Color bord = active ? item.AccentColor : Color.FromArgb(30, 0, 0, 0);
        int bw = active ? 2 : 1;

        FillRoundRect(g, bg, r, 10);
        using (Pen pen = new Pen(bord, bw))
        {
            DrawRoundRectPen(g, pen, r, 10);
        }

        Rectangle badge = new Rectangle(r.Right - 30, r.Y + 5, 26, 16);
        FillRoundRect(g, active ? item.AccentColor : GRAY_LIGHT, badge, 8);
        g.DrawString("M" + item.MarkerID, fontMarker,
                     new SolidBrush(active ? Color.White : GRAY_MID),
                     new PointF(badge.X + 3, badge.Y + 1));

        DrawEmoji(g, item.Emoji, new Rectangle(r.X + 6, r.Y + 2, 26, 26));
        g.DrawString(item.Name, fontH3, new SolidBrush(TEXT_PRI), new PointF(r.X + 6, r.Y + 32));

        if (item.IsTool)
        {
            g.DrawString(active ? "Physical Object Ready" : "Tool not scanned",
                         fontSmall,
                         new SolidBrush(active ? item.AccentColor : TEXT_SEC),
                         new PointF(r.X + 6, r.Y + 58));
        }
        else if (active)
        {
            float angle = GetMarkerAngle(item.MarkerID, obj);
            int grams = item.GetGrams(angle);
            int kcal = item.GetCalories(angle);
            string sourceText = obj != null ? "" : "  •  YOLO";
            g.DrawString(grams + "g  •  " + kcal + " kcal" + sourceText,
                         fontSmall,
                         new SolidBrush(item.AccentColor),
                         new PointF(r.X + 6, r.Y + 58));

            if (obj != null)
            {
                float deg = GetMarkerAngle(item.MarkerID, obj) * 180f / (float)Math.PI;
                if (deg < 0) deg += 360f;
                DrawRotArc(g, new Point(r.Right - 18, r.Bottom - 18), 12, deg, item.AccentColor);
            }
        }
        else
        {
            g.DrawString(item.KcalPerGram.ToString("F1") + " kcal/g",
                         fontSmall,
                         new SolidBrush(TEXT_SEC),
                         new PointF(r.X + 6, r.Y + 58));
        }
    }

    private void DrawRotationLegend(Graphics g, Rectangle r)
    {
        FillRoundRect(g, AMBER_LIGHT, r, CARD_R);
        g.DrawString("Rotation → Quantity", fontH3, new SolidBrush(AMBER), new PointF(r.X + 10, r.Y + 8));

        string[] labels = new string[]
        {
            "0°–89° = Small",
            "90°–179° = Medium",
            "180°–269° = Large",
            "270°–359° = X-Large"
        };

        for (int i = 0; i < labels.Length; i++)
        {
            int col = i % 2;
            int row = i / 2;
            g.DrawString(labels[i], fontSmall, new SolidBrush(TEXT_PRI),
                         new PointF(r.X + 10 + col * 120, r.Y + 30 + row * 18));
        }
    }

    private void DrawPhysicalObjectPanel(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        FillRoundRect(g, BLUE_LIGHT, r, CARD_R);

        bool spoon = detected.ContainsKey(6);
        bool pot = detected.ContainsKey(7);
        bool knife = detected.ContainsKey(8);

        g.DrawString("Physical Objects", fontH3, new SolidBrush(BLUE), new PointF(r.X + 10, r.Y + 8));
        g.DrawString("🥄 Spoon: " + (spoon ? "Scanned" : "Missing"), fontBody, new SolidBrush(TEXT_PRI), new PointF(r.X + 10, r.Y + 34));
        g.DrawString("🍲 Pot: " + (pot ? "Scanned" : "Missing"), fontBody, new SolidBrush(TEXT_PRI), new PointF(r.X + 10, r.Y + 56));
        g.DrawString("🔪 Knife: " + (knife ? "Scanned" : "Missing"), fontBody, new SolidBrush(TEXT_PRI), new PointF(r.X + 10, r.Y + 78));

        if (activeUserProfile != null)
        {
            g.DrawString("Bio: " + currentBio, fontTiny, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 10, r.Y + 98, r.Width - 20, 18));
        }
    }

    private void DrawCenterPanel(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        int x = 550;
        int y = PAD + 132 + 16;
        int w = 800;

        DrawTotalCalCard(g, new Rectangle(x, y, w, 80), GetTotalCalories(detected), detected.Count);

        Rectangle menuCard = new Rectangle(x, y + 95, w, 515);
        DrawCircularMenuCard(g, detected, menuCard);

        Rectangle detailCard = new Rectangle(x + 120, y + 626, 275, 178);
        DrawCircularMenuDetailCard(g, detected, detailCard);

        Rectangle selectedCard = new Rectangle(x + 515, y + 626, 150, 178);
        FillRoundRect(g, Color.FromArgb(241, 239, 232), selectedCard, 14);
        g.DrawString("Current", fontSmall, new SolidBrush(TEXT_SEC), new PointF(selectedCard.X + 40, selectedCard.Y + 22));
        g.DrawString(circularMenuItems[(int)currentMenuView], fontH2, new SolidBrush(BLUE), new RectangleF(selectedCard.X + 40, selectedCard.Y + 50, selectedCard.Width - 50, 36));
        g.DrawString("Selected", fontSmall, new SolidBrush(TEXT_SEC), new PointF(selectedCard.X + 40, selectedCard.Y + 105));
        g.DrawString(circularMenuItems[(int)confirmedMenuView], fontH2, new SolidBrush(GREEN), new RectangleF(selectedCard.X + 40, selectedCard.Y + 132, selectedCard.Width - 50, 36));
    }

    private void DrawCircularMenuCard(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        StringFormat center = new StringFormat();
        center.Alignment = StringAlignment.Center;

        g.DrawString("TUIO Circular Menu", fontH2, new SolidBrush(TEXT_PRI), new RectangleF(r.X, r.Y, r.Width, 24), center);
        g.DrawString("Marker 20 rotates between views. Hold steady to confirm selection.", fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(r.X, r.Y + 28, r.Width, 22), center);

        Rectangle wheelRect = new Rectangle(r.X + 155, r.Y + 62, 490, 490);
        TuioDemoObject menuObj = GetLiveMenuControllerObject();
        DrawCircularMenuWheel(g, wheelRect, menuObj);

        string status = menuStatusText;
        g.DrawString(status, fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 70, r.Bottom - 35, r.Width - 140, 34), center);
    }

    private void DrawCircularMenuWheel(Graphics g, Rectangle r, TuioDemoObject menuObj)
    {
        int cx = r.X + r.Width / 2;
        int cy = r.Y + r.Height / 2;
        int outerRadius = Math.Min(r.Width, r.Height) / 2;
        int innerRadius = 92;
        int sliceCount = circularMenuItems.Length;
        float sweep = 360f / sliceCount;
        float startBase = -120f;

        Color[] fills = new Color[] { GREEN_LIGHT, AMBER_LIGHT, CORAL_LIGHT, BLUE_LIGHT, Color.FromArgb(236, 247, 229), PURPLE_LIGHT };
        Color[] accents = new Color[] { GREEN, Color.FromArgb(235, 133, 18), CORAL, BLUE, Color.FromArgb(93, 158, 39), PURPLE };
        string[] icons = new string[] { "⌂", "📖", "🔥", "☑", "♥", "💡" };

        using (Pen shadow = new Pen(Color.FromArgb(22, 0, 0, 0), 10f))
            g.DrawEllipse(shadow, r.X + 4, r.Y + 6, r.Width - 8, r.Height - 8);

        for (int i = 0; i < sliceCount; i++)
        {
            bool isCurrent = i == (int)currentMenuView;
            bool isConfirmed = i == (int)confirmedMenuView;
            Color fill = fills[i];
            if (!isCurrent && !isConfirmed) fill = Color.FromArgb(235, fill);

            using (GraphicsPath slicePath = new GraphicsPath())
            {
                slicePath.AddPie(r, startBase + i * sweep, sweep - 1.5f);
                using (SolidBrush br = new SolidBrush(fill))
                    g.FillPath(br, slicePath);
                using (Pen pen = new Pen(Color.White, 4f))
                    g.DrawPath(pen, slicePath);
                if (isCurrent || isConfirmed)
                {
                    using (Pen pen = new Pen(accents[i], isConfirmed ? 3.5f : 2.2f))
                        g.DrawPath(pen, slicePath);
                }
            }

            float mid = (startBase + i * sweep + sweep / 2f) * (float)Math.PI / 180f;
            int labelRadius = (innerRadius + outerRadius) / 2 + 6;
            float tx = cx + (float)Math.Cos(mid) * labelRadius;
            float ty = cy + (float)Math.Sin(mid) * labelRadius;

            SizeF iconSz = g.MeasureString(icons[i], fontEmoji);
            g.DrawString(icons[i], fontEmoji, new SolidBrush(accents[i]), tx - iconSz.Width / 2f, ty - 30);
            SizeF sz = g.MeasureString(circularMenuItems[i], fontH3);
            g.DrawString(circularMenuItems[i], fontH3, new SolidBrush(TEXT_PRI), tx - sz.Width / 2f, ty + 10);
        }

        using (SolidBrush centerBrush = new SolidBrush(Color.White))
            g.FillEllipse(centerBrush, cx - innerRadius, cy - innerRadius, innerRadius * 2, innerRadius * 2);
        using (Pen centerPen = new Pen(Color.FromArgb(45, 0, 0, 0), 1.5f))
            g.DrawEllipse(centerPen, cx - innerRadius, cy - innerRadius, innerRadius * 2, innerRadius * 2);

        SizeF msz = g.MeasureString("MENU", fontTitle);
        g.DrawString("MENU", fontTitle, new SolidBrush(TEXT_PRI), new PointF(cx - msz.Width / 2f, cy - msz.Height / 2f));

        if (menuObj != null)
        {
            float deg = NormalizeDegrees(menuObj.Angle);
            float rad = (deg - 90f) * (float)Math.PI / 180f;
            int ex = cx + (int)(Math.Cos(rad) * (outerRadius - 16));
            int ey = cy + (int)(Math.Sin(rad) * (outerRadius - 16));
            using (Pen arrowPen = new Pen(BLUE, 4f))
                g.DrawLine(arrowPen, cx, cy, ex, ey);
            g.FillEllipse(new SolidBrush(BLUE), ex - 7, ey - 7, 14, 14);
        }
    }

    private void DrawCircularMenuDetailCard(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        FillRoundRect(g, CARD_BG, r, CARD_R);
        DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), r, CARD_R);

        string title = circularMenuItems[(int)confirmedMenuView] + " View";
        g.DrawString(title, fontH3, new SolidBrush(TEXT_PRI), new PointF(r.X + 12, r.Y + 10));

        switch (confirmedMenuView)
        {
            case MenuView.Overview:
                DrawOverviewDetail(g, detected, r);
                break;
            case MenuView.Recipes:
                DrawRecipesDetail(g, detected, r);
                break;
            case MenuView.Steps:
                DrawStepsDetail(g, detected, r);
                break;
            case MenuView.Health:
                DrawHealthDetail(g, detected, r);
                break;
            case MenuView.Calories:
                DrawCaloriesDetail(g, detected, r);
                break;
            case MenuView.Tips:
                DrawTipsDetail(g, detected, r);
                break;
        }
    }

    private void DrawOverviewDetail(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        Rectangle inner = new Rectangle(r.X + 12, r.Y + 38, r.Width - 24, r.Height - 50);
        FillRoundRect(g, Color.FromArgb(248, 252, 249), inner, 16);
        DrawRoundRect(g, Color.FromArgb(28, GREEN), inner, 16);

        int score = CalculateHealthScore(detected);
        int totalCalories = GetTotalCalories(detected);
        int scanned = detected == null ? 0 : detected.Count;
        string mode = IsClientMode() ? "Client ready-meal dashboard" : "Chef ingredient dashboard";

        DrawEmoji(g, "📊", new Rectangle(inner.X + 12, inner.Y + 12, 28, 28));
        g.DrawString("Overview Summary", fontH3, new SolidBrush(TEXT_PRI), new PointF(inner.X + 48, inner.Y + 12));
        g.DrawString(mode, fontTiny, new SolidBrush(TEXT_SEC), new PointF(inner.X + 50, inner.Y + 34));

        Rectangle kcalBox = new Rectangle(inner.X + 14, inner.Y + 62, 92, 58);
        Rectangle scoreBox = new Rectangle(kcalBox.Right + 10, inner.Y + 62, 92, 58);
        FillRoundRect(g, GREEN_LIGHT, kcalBox, 14);
        FillRoundRect(g, BLUE_LIGHT, scoreBox, 14);
        g.DrawString(totalCalories.ToString(), new Font("Segoe UI", 15, FontStyle.Bold), new SolidBrush(GREEN), new PointF(kcalBox.X + 10, kcalBox.Y + 8));
        g.DrawString("kcal", fontTiny, new SolidBrush(TEXT_SEC), new PointF(kcalBox.X + 12, kcalBox.Y + 34));
        g.DrawString(score.ToString() + "/100", new Font("Segoe UI", 15, FontStyle.Bold), new SolidBrush(BLUE), new PointF(scoreBox.X + 8, scoreBox.Y + 8));
        g.DrawString("health", fontTiny, new SolidBrush(TEXT_SEC), new PointF(scoreBox.X + 12, scoreBox.Y + 34));

        string summary;
        if (scanned == 0)
            summary = IsClientMode() ? "Scan ready-meal markers to compare calories, protein, and health score." : " ";
        else
            summary = scanned + " scanned marker(s). Rotate markers or move YOLO object position to adjust quantity.";

        g.DrawString(summary, fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(inner.X + 14, inner.Y + 132, inner.Width - 28, 38));

        string focus = " " + circularMenuItems[(int)confirmedMenuView] + " ";
        g.DrawString(focus, fontTiny, new SolidBrush(currentAccent), new RectangleF(inner.X + 14, inner.Bottom - 28, inner.Width - 28, 20));
    }

    private void DrawRecipesDetail(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        HashSet<int> activeMarkers = new HashSet<int>(detected.Keys);
        Recipe best = null;
        List<int> bestHave = new List<int>();
        List<int> bestMissing = new List<int>();

        foreach (Recipe recipe in GetPersonalizedRecipes())
        {
            List<int> have = recipe.RequiredMarkers.Where(m => activeMarkers.Contains(m)).ToList();
            if (have.Count == 0) continue;
            List<int> missing = recipe.RequiredMarkers.Where(m => !activeMarkers.Contains(m)).ToList();
            if (best == null || have.Count > bestHave.Count)
            {
                best = recipe;
                bestHave = have;
                bestMissing = missing;
            }
        }

        if (best == null)
        {
            g.DrawString("Scan at least one allowed ingredient to start getting personalized recipe suggestions.", fontBody, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 12, r.Y + 48, r.Width - 24, 60));
            return;
        }

        DrawRecipeCard(g, best, new Rectangle(r.X + 12, r.Y + 42, r.Width - 24, 106), bestHave, bestMissing);
    }

    private void DrawStepsDetail(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        Rectangle inner = new Rectangle(r.X + 12, r.Y + 40, r.Width - 24, r.Height - 52);
        if (showMealSteps)
            DrawMealStepsCard(g, detected, inner);
        else
            DrawToggleInfoCard(g, inner, "Meal steps are hidden. Press M to show them.");
    }

    private void DrawHealthDetail(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        int score = CalculateHealthScore(detected);
        int totalCalories = GetTotalCalories(detected);
        g.DrawString("Current meal score: " + score + "/100", fontH2, new SolidBrush(GREEN), new PointF(r.X + 12, r.Y + 44));
        g.DrawString("Calories: " + totalCalories + " kcal", fontBody, new SolidBrush(TEXT_PRI), new PointF(r.X + 12, r.Y + 74));
        g.DrawString("Protein and vegetables raise the score. Too much oil or very high calories reduce it.", fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 12, r.Y + 102, r.Width - 24, 42));
    }

    private void DrawCaloriesDetail(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        int y = r.Y + 42;
        int shown = 0;
        foreach (KeyValuePair<int, TuioDemoObject> kv in detected)
        {
            KitchenItem item = items.FirstOrDefault(i => i.MarkerID == kv.Key);
            if (item == null || item.IsTool) continue;
            g.DrawString(item.Name + ": " + item.GetCalories(GetMarkerAngle(kv.Key, kv.Value)) + " kcal", fontBody, new SolidBrush(item.AccentColor), new PointF(r.X + 12, y));
            y += 24;
            shown++;
            if (y > r.Bottom - 36) break;
        }
        if (shown == 0)
            g.DrawString("No ingredient calories yet. Scan food items first.", fontBody, new SolidBrush(TEXT_SEC), new PointF(r.X + 12, y));

        g.DrawString("Total: " + GetTotalCalories(detected) + " kcal", fontH2, new SolidBrush(GREEN), new PointF(r.X + 12, r.Bottom - 34));
    }

    private void DrawTipsDetail(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        Rectangle inner = new Rectangle(r.X + 12, r.Y + 40, r.Width - 24, r.Height - 52);
        if (showHealthTips)
            DrawTipsPanel(g, detected, inner);
        else
            DrawToggleInfoCard(g, inner, "Health tips are hidden in this view.");
    }

    private void DrawHealthScoreCard(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        int totalCalories = GetTotalCalories(detected);
        int score = CalculateHealthScore(detected);

        FillRoundRect(g, GREEN_LIGHT, r, CARD_R);
        g.DrawString("Health Score", fontH3, new SolidBrush(TEAL_DARK), new PointF(r.X + 12, r.Y + 8));
        g.DrawString(score.ToString() + "/100", fontBig, new SolidBrush(GREEN), new PointF(r.X + 12, r.Y + 28));
        g.DrawString("Total Calories: " + totalCalories + " kcal", fontBody, new SolidBrush(TEXT_SEC), new PointF(r.X + 160, r.Y + 40));

        string level = "Balanced";
        if (score >= 85) level = "Very Healthy";
        else if (score >= 65) level = "Healthy";
        else if (score >= 45) level = "Needs Improvement";
        else level = "High Calorie";

        g.DrawString("Status: " + level, fontBody, new SolidBrush(TEXT_PRI), new PointF(r.X + 160, r.Y + 62));
    }

    private void DrawMealStepsCard(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle r)
    {
        FillRoundRect(g, CARD_BG, r, CARD_R);
        DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), r, CARD_R);

        int pad = 12;
        g.DrawString("Cooking Steps", fontH3, new SolidBrush(TEXT_PRI), new PointF(r.X + pad, r.Y + 8));

        Recipe readyRecipe = GetReadyRecipe(detected);

        if (readyRecipe == null)
        {
            currentRecipeName = "";
            currentRecipeStepIndex = 0;

            Rectangle emptyBox = new Rectangle(r.X + pad, r.Y + 42, r.Width - pad * 2, Math.Max(76, r.Height - 78));
            FillRoundRect(g, GRAY_LIGHT, emptyBox, 10);
            g.DrawString("No complete personalized recipe yet.", fontBody, new SolidBrush(TEXT_PRI), new PointF(emptyBox.X + 12, emptyBox.Y + 12));
            g.DrawString("Scan the required ingredients and tools first. Then use hand gesture RIGHT for next step and LEFT for previous step.",
                fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(emptyBox.X + 12, emptyBox.Y + 38, emptyBox.Width - 24, emptyBox.Height - 44));
            g.DrawString(lastGestureCommand, fontTiny, new SolidBrush(TEXT_SEC), new PointF(r.X + pad, r.Bottom - 22));
            return;
        }

        if (!string.Equals(currentRecipeName, readyRecipe.Name, StringComparison.OrdinalIgnoreCase))
        {
            currentRecipeName = readyRecipe.Name;
            currentRecipeStepIndex = 0;
        }

        if (currentRecipeStepIndex < 0) currentRecipeStepIndex = 0;
        if (currentRecipeStepIndex >= readyRecipe.Steps.Length) currentRecipeStepIndex = readyRecipe.Steps.Length - 1;

        Rectangle recipeHeader = new Rectangle(r.X + pad, r.Y + 38, r.Width - pad * 2, 48);
        FillRoundRect(g, Color.FromArgb(248, 248, 248), recipeHeader, 10);
        DrawEmoji(g, readyRecipe.Emoji, new Rectangle(recipeHeader.X + 10, recipeHeader.Y + 8, 28, 28));
        g.DrawString(readyRecipe.Name, fontBody, new SolidBrush(TEXT_PRI), new RectangleF(recipeHeader.X + 46, recipeHeader.Y + 8, recipeHeader.Width - 56, 18));
        g.DrawString(readyRecipe.Category + "  •  RIGHT next  •  LEFT previous", fontTiny, new SolidBrush(TEXT_SEC), new RectangleF(recipeHeader.X + 46, recipeHeader.Y + 28, recipeHeader.Width - 56, 16));

        Rectangle focus = new Rectangle(r.X + pad, recipeHeader.Bottom + 10, r.Width - pad * 2, 92);
        FillRoundRect(g, currentAccentLight, focus, 12);
        g.DrawString("Current Step", fontH3, new SolidBrush(currentAccent), new PointF(focus.X + 12, focus.Y + 10));
        g.DrawString((currentRecipeStepIndex + 1).ToString() + "/" + readyRecipe.Steps.Length, fontH2, new SolidBrush(TEXT_PRI), new PointF(focus.Right - 60, focus.Y + 12));
        g.DrawString(readyRecipe.Steps[currentRecipeStepIndex], fontBody, new SolidBrush(TEXT_PRI), new RectangleF(focus.X + 12, focus.Y + 42, focus.Width - 24, 42));

        int stepY = focus.Bottom + 10;
        int maxRows = Math.Max(1, (r.Bottom - stepY - 28) / 28);
        for (int i = 0; i < readyRecipe.Steps.Length && i < maxRows; i++)
        {
            Rectangle row = new Rectangle(r.X + pad, stepY, r.Width - pad * 2, 24);
            if (i == currentRecipeStepIndex)
                FillRoundRect(g, currentAccentLight, row, 8);

            string stepText = (i + 1).ToString() + ". " + readyRecipe.Steps[i];
            Brush textBrush = (i == currentRecipeStepIndex) ? new SolidBrush(currentAccent) : new SolidBrush(TEXT_PRI);
            g.DrawString(stepText, fontTiny, textBrush, new RectangleF(row.X + 8, row.Y + 5, row.Width - 16, row.Height - 4));
            stepY += 28;
        }

        g.DrawString(lastGestureCommand, fontTiny, new SolidBrush(TEXT_SEC), new PointF(r.X + pad, r.Bottom - 22));
    }

    private void DrawToggleInfoCard(Graphics g, Rectangle r, string text)
    {
        FillRoundRect(g, GRAY_LIGHT, r, CARD_R);
        g.DrawString(text, fontBody, new SolidBrush(TEXT_SEC), new PointF(r.X + 12, r.Y + 20));
    }

    private void DrawRightPanel(Graphics g, Dictionary<int, TuioDemoObject> detected)
    {
        int x = 1380;
        int startY = PAD + 132 + 110;
        int w = WIN_W - x - 58;

        if (w < 420)
        {
            x = WIN_W - 500;
            w = 460;
        }

        Rectangle crudRect = new Rectangle(x, startY, w, 210);
        FillRoundRect(g, Color.FromArgb(229, 244, 255), crudRect, CARD_R);
        using (Pen pen = new Pen(BLUE, 2))
            DrawRoundRectPen(g, pen, crudRect, CARD_R);

        g.DrawString("✨ Context Content Studio", fontH3, new SolidBrush(TEXT_PRI), new PointF(crudRect.X + 18, crudRect.Y + 22));
        g.DrawString("Tangible CRUD using TUIO markers", fontSmall, new SolidBrush(TEXT_SEC), new PointF(crudRect.X + 18, crudRect.Y + 56));

        DrawCrudActionChip(g, new Rectangle(crudRect.X + 18, crudRect.Y + 92, 102, 26), "M11", "Add", GREEN);
        DrawCrudActionChip(g, new Rectangle(crudRect.X + 126, crudRect.Y + 92, 116, 26), "M12", "Update", BLUE);
        DrawCrudActionChip(g, new Rectangle(crudRect.X + 248, crudRect.Y + 92, 112, 26), "M13", "Delete", CORAL);
        DrawCrudActionChip(g, new Rectangle(crudRect.X + 366, crudRect.Y + 92, 112, 26), "M14", "Reload", AMBER);

        string selectedLine = string.IsNullOrWhiteSpace(selectedContentRecipeName) ? "Selected: nearest matching context recipe" : "Selected: " + selectedContentRecipeName;
        g.DrawString(selectedLine, fontSmall, new SolidBrush(BLUE), new RectangleF(crudRect.X + 18, crudRect.Y + 142, crudRect.Width - 36, 20));
        g.DrawString(crudStatusText, fontTiny, new SolidBrush(TEXT_SEC), new RectangleF(crudRect.X + 18, crudRect.Y + 172, crudRect.Width - 36, 28));
        startY += 230;

        HashSet<int> activeMarkers = new HashSet<int>(detected.Keys);
        int drawn = 0;
        int maxRecipeCards = 2;
        if (emotionHappyMode) maxRecipeCards = 4;
        if (emotionSadMode || emotionAngryMode) maxRecipeCards = 1;

        foreach (Recipe recipe in GetPersonalizedRecipes())
        {
            List<int> have = recipe.RequiredMarkers.Where(m => activeMarkers.Contains(m)).ToList();
            List<int> missing = recipe.RequiredMarkers.Where(m => !activeMarkers.Contains(m)).ToList();
            if (have.Count == 0) continue;

            int cardH = missing.Count > 0 ? 120 : 104;
            if (emotionHappyMode) cardH += 36;
            DrawRecipeCard(g, recipe, new Rectangle(x, startY, w, cardH), have, missing);
            startY += cardH + (emotionHappyMode ? 14 : 12);
            drawn++;
            if (drawn >= maxRecipeCards) break;
        }

        if (drawn == 0)
        {
            Rectangle er = new Rectangle(x, startY, w, 178);
            FillRoundRect(g, CARD_BG, er, CARD_R);
            DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), er, CARD_R);
            g.DrawString("Meal Suggestions", fontH2, new SolidBrush(TEXT_PRI), new PointF(er.X + 18, er.Y + 22));
            g.DrawString("Scan ingredients and physical objects to see", fontSmall, new SolidBrush(TEXT_SEC), new PointF(er.X + 18, er.Y + 64));
            g.DrawString("personalized meal suggestions.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(er.X + 18, er.Y + 94));
            DrawEmoji(g, "🍽", new Rectangle(er.Right - 140, er.Y + 55, 96, 96));
            startY += 196;
        }

        if (emotionSadMode || emotionAngryMode)
        {
            startY += 8;
            Rectangle calmBox = new Rectangle(x, startY, w, 74);
            FillRoundRect(g, currentAccentLight, calmBox, CARD_R);
            string msg = emotionSadMode
                ? "Calm Mode: Showing one simple recommendation only."
                : "Help Mode: Interface simplified to reduce frustration.";
            g.DrawString(msg, fontH3, new SolidBrush(currentAccent),
                new RectangleF(calmBox.X + 14, calmBox.Y + 20, calmBox.Width - 28, calmBox.Height - 20));
        }
        else if (showHealthTips)
        {
            startY += 8;
            DrawTipsPanel(g, detected, new Rectangle(x, startY, w, Math.Min(150, WIN_H - startY - 86)));
        }
        else
        {
            startY += 8;
            Rectangle er = new Rectangle(x, startY, w, 70);
            FillRoundRect(g, GRAY_LIGHT, er, CARD_R);
            g.DrawString("Health tips are hidden in this view.", fontBody, new SolidBrush(TEXT_SEC), new PointF(er.X + 14, er.Y + 24));
        }
    }

    private void DrawCrudActionChip(Graphics g, Rectangle r, string marker, string action, Color color)
    {
        FillRoundRect(g, Color.White, r, 11);
        using (Pen pen = new Pen(color, 1))
        {
            DrawRoundRectPen(g, pen, r, 11);
        }

        Rectangle markerBox = new Rectangle(r.X + 4, r.Y + 3, 34, r.Height - 6);
        FillRoundRect(g, color, markerBox, 8);
        g.DrawString(marker, fontTiny, new SolidBrush(Color.White), new PointF(markerBox.X + 5, markerBox.Y + 3));
        g.DrawString(action, fontTiny, new SolidBrush(TEXT_PRI), new PointF(r.X + 43, r.Y + 5));
    }

    private void DrawPersonalizationCard(Graphics g, Rectangle r)
    {
        FillRoundRect(g, currentAccentLight, r, CARD_R);
        DrawRoundRect(g, currentAccent, r, CARD_R);

        g.DrawString("Personalized Profile", fontH3, new SolidBrush(currentAccent), new PointF(r.X + 12, r.Y + 8));
        g.DrawString("User: " + currentUserName + "  •  Age: " + activeUserProfile.Age, fontBody, new SolidBrush(TEXT_PRI), new PointF(r.X + 12, r.Y + 32));

        string fav = activeUserProfile.FavoriteMeals.Count > 0
            ? string.Join(", ", activeUserProfile.FavoriteMeals.ToArray())
            : "No favorites specified";

        g.DrawString("Favorites: " + fav, fontSmall, new SolidBrush(TEXT_SEC), new RectangleF(r.X + 12, r.Y + 54, r.Width - 24, 18));
    }

    private void DrawTipsPanel(Graphics g, Dictionary<int, TuioDemoObject> detected, Rectangle area)
    {
        FillRoundRect(g, Color.FromArgb(255, 249, 239), area, CARD_R);
        DrawRoundRect(g, Color.FromArgb(45, 224, 174, 82), area, CARD_R);
        g.DrawString("Diet Tips", fontH2, new SolidBrush(TEXT_PRI), new PointF(area.X + 18, area.Y + 22));

        List<string> tips = new List<string>();
        foreach (int mid in detected.Keys)
        {
            KitchenItem item = items.FirstOrDefault(i => i.MarkerID == mid);
            if (item != null && item.Tips != null)
            {
                foreach (string tip in item.Tips)
                {
                    if (!tips.Contains(tip)) tips.Add(tip);
                    if (tips.Count >= 3) break;
                }
            }
            if (tips.Count >= 3) break;
        }

        if (activeUserProfile != null && !string.IsNullOrWhiteSpace(activeUserProfile.Dislikes))
        {
            string p = "Profile preference: avoid " + activeUserProfile.Dislikes + ".";
            if (!tips.Contains(p)) tips.Insert(0, p);
        }

        if (tips.Count == 0)
        {
            g.DrawString("Scan some items to get health advice.", fontSmall, new SolidBrush(TEXT_SEC), new PointF(area.X + 18, area.Y + 70));
            DrawEmoji(g, "💡", new Rectangle(area.Right - 118, area.Y + 48, 88, 88));
            return;
        }

        int y = area.Y + 58;
        foreach (string tip in tips)
        {
            if (y + 24 > area.Bottom - 12) break;
            g.DrawString("💡 " + tip, fontSmall, new SolidBrush(TEXT_PRI), new RectangleF(area.X + 18, y, area.Width - 36, 24));
            y += 28;
        }
    }

    private int GetTotalCalories(Dictionary<int, TuioDemoObject> detected)
    {
        int totalCal = 0;
        foreach (KeyValuePair<int, TuioDemoObject> kv in detected)
        {
            KitchenItem item = items.FirstOrDefault(i => i.MarkerID == kv.Key);
            if (item == null || item.IsTool) continue;
            float angle = GetMarkerAngle(kv.Key, kv.Value);
            totalCal += item.GetCalories(angle);
        }
        return totalCal;
    }

    private int CalculateHealthScore(Dictionary<int, TuioDemoObject> detected)
    {
        int score = 50;

        bool hasChicken = detected.ContainsKey(0);
        bool hasTomato = detected.ContainsKey(1);
        bool hasOnion = detected.ContainsKey(2);
        bool hasSpices = detected.ContainsKey(3);
        bool hasRice = detected.ContainsKey(4);
        bool hasOil = detected.ContainsKey(5);

        if (hasChicken) score += 15;
        if (hasTomato) score += 10;
        if (hasOnion) score += 10;
        if (hasSpices) score += 5;
        if (hasRice) score += 5;

        if (hasOil)
        {
            KitchenItem oil = items.FirstOrDefault(i => i.MarkerID == 5);
            TuioDemoObject oilObj = detected[5];
            float oilAngle = GetMarkerAngle(5, oilObj);
            if (oil != null)
            {
                int oilGrams = oil.GetGrams(oilAngle);
                if (oilGrams <= 5) score += 5;
                else if (oilGrams <= 10) score -= 5;
                else score -= 15;
            }
        }

        int totalCalories = GetTotalCalories(detected);
        if (totalCalories > 800) score -= 15;
        else if (totalCalories > 500) score -= 5;

        if (score < 0) score = 0;
        if (score > 100) score = 100;
        return score;
    }

    private void DrawTotalCalCard(Graphics g, Rectangle r, int total, int count)
    {
        using (LinearGradientBrush br = new LinearGradientBrush(r, Color.FromArgb(221, 248, 240), Color.FromArgb(231, 249, 244), 0f))
            g.FillPath(br, RoundedRect(r, CARD_R));

        g.DrawString(total.ToString(), fontBig, new SolidBrush(GREEN), new PointF(r.X + 22, r.Y + 12));
        g.DrawString("total kcal", fontSmall, new SolidBrush(TEAL_DARK), new PointF(r.X + 22, r.Y + 50));
        g.DrawString(count.ToString() + " scanned markers", fontSmall, new SolidBrush(TEXT_PRI), new PointF(r.X + r.Width - 170, r.Y + 28));
        if (blockedMarkers.Count > 0)
        {
            g.DrawString("Hidden markers for profile: " + string.Join(", ", blockedMarkers.Select(x => "M" + x).ToArray()),
                fontTiny, new SolidBrush(currentAccent), new PointF(r.X + 22, r.Y + 66));
        }
    }

    private void DrawRecipeCard(Graphics g, Recipe recipe, Rectangle r, List<int> have, List<int> missing)
    {
        bool canMake = missing.Count == 0;
        bool favorite = activeUserProfile != null &&
                        activeUserProfile.FavoriteMeals.Any(f =>
                            string.Equals(f, recipe.Name, StringComparison.OrdinalIgnoreCase));

        bool selectedContent = recipe.IsContentRecipe && !string.IsNullOrWhiteSpace(selectedContentRecipeName) &&
                               string.Equals(recipe.Name, selectedContentRecipeName, StringComparison.OrdinalIgnoreCase);

        Color bg = canMake ? GREEN_LIGHT : CARD_BG;
        Color border = selectedContent ? BLUE : (favorite ? currentAccent : (canMake ? GREEN : Color.FromArgb(30, 0, 0, 0)));
        int bw = selectedContent ? 3 : (favorite || canMake ? 2 : 1);

        FillRoundRect(g, bg, r, CARD_R);
        using (Pen pen = new Pen(border, bw))
        {
            DrawRoundRectPen(g, pen, r, CARD_R);
        }

        DrawEmoji(g, recipe.Emoji, new Rectangle(r.X + 8, r.Y + 6, 26, 26));
        g.DrawString(recipe.Name, fontH3, new SolidBrush(TEXT_PRI), new PointF(r.X + 36, r.Y + 10));
        g.DrawString(recipe.Category + (favorite ? "  •  Favorite" : ""), fontTiny, new SolidBrush(favorite ? currentAccent : TEXT_SEC), new PointF(r.X + 36, r.Y + 32));

        string statusText = selectedContent ? "Selected" : (canMake ? "Ready to Cook" : "Missing " + missing.Count);
        Color statusBg = selectedContent ? BLUE : (canMake ? GREEN : (favorite ? currentAccent : AMBER));
        Rectangle badgeRect = new Rectangle(r.Right - 110, r.Y + 8, 98, 20);
        FillRoundRect(g, statusBg, badgeRect, 10);
        g.DrawString(statusText, fontTiny, new SolidBrush(Color.White), new PointF(badgeRect.X + 8, badgeRect.Y + 4));

        int tx = r.X + 8;
        int ty = r.Y + 54;

        foreach (int mid in recipe.RequiredMarkers)
        {
            KitchenItem item = items.FirstOrDefault(i => i.MarkerID == mid);
            if (item == null) continue;

            bool h = have.Contains(mid);
            string tagText = item.Emoji + " " + item.Name;
            int tw = (int)g.MeasureString(tagText, fontTiny).Width + 12;
            Rectangle tagR = new Rectangle(tx, ty, tw, 18);

            FillRoundRect(g, h ? GREEN_LIGHT : CORAL_LIGHT, tagR, 9);
            g.DrawString(tagText, fontTiny,
                         new SolidBrush(h ? TEAL_DARK : CORAL),
                         new PointF(tx + 4, ty + 2));

            tx += tw + 4;
            if (tx > r.Right - 110)
            {
                tx = r.X + 8;
                ty += 22;
            }
        }

        if (!canMake)
        {
            string miss = "Missing: " + string.Join(", ",
                missing.Select(m =>
                {
                    KitchenItem item = items.FirstOrDefault(i => i.MarkerID == m);
                    return item != null ? item.Name : "?";
                }).ToArray());

            g.DrawString(miss, fontSmall, new SolidBrush(CORAL), new PointF(r.X + 8, r.Bottom - 22));
        }
    }

    private void DrawFooterMappingBar(Graphics g)
    {
        int w = 1110;
        int h = 50;
        Rectangle r = new Rectangle((WIN_W - w) / 2, WIN_H - h - 18, w, h);
        FillRoundRect(g, Color.FromArgb(244, 240, 233), r, 10);
        DrawRoundRect(g, Color.FromArgb(30, 0, 0, 0), r, 10);

        string text = IsClientMode()
            ? "Client marker mapping: 0=Pizza  1=Pasta  2=Burger  3=Sushi Bowl  4=Chicken Wrap  5=Salad  6=Pancakes  7=Smoothie  8=Dessert"
            : "Marker mapping:    0=Chicken    1=Tomato    2=Onion    3=Spices    4=Rice    5=Oil    6=Spoon    7=Pot    8=Knife";
        g.DrawString(text, fontSmall, new SolidBrush(TEXT_PRI), new PointF(r.X + 22, r.Y + 16));
        g.DrawString("|", fontSmall, new SolidBrush(TEXT_SEC), new PointF(r.X + 720, r.Y + 16));
        g.DrawString("👤  Login Marker: 25", fontSmall, new SolidBrush(TEXT_PRI), new PointF(r.X + 760, r.Y + 16));
        g.DrawString("◯  Menu Marker: 20", fontSmall, new SolidBrush(TEXT_PRI), new PointF(r.X + 935, r.Y + 16));
    }

    // =========================
    // Drawing helpers
    // =========================
    private void FillRoundRect(Graphics g, Color color, Rectangle r, int radius)
    {
        using (GraphicsPath path = RoundedRect(r, radius))
        using (SolidBrush brush = new SolidBrush(color))
        {
            g.FillPath(brush, path);
        }
    }

    private void DrawRoundRect(Graphics g, Color color, Rectangle r, int radius)
    {
        using (GraphicsPath path = RoundedRect(r, radius))
        using (Pen pen = new Pen(color, 1))
        {
            g.DrawPath(pen, path);
        }
    }

    private void DrawRoundRectPen(Graphics g, Pen pen, Rectangle r, int radius)
    {
        using (GraphicsPath path = RoundedRect(r, radius))
        {
            g.DrawPath(pen, path);
        }
    }

    private GraphicsPath RoundedRect(Rectangle r, int radius)
    {
        int d = radius * 2;
        GraphicsPath path = new GraphicsPath();
        path.AddArc(r.X, r.Y, d, d, 180, 90);
        path.AddArc(r.Right - d, r.Y, d, d, 270, 90);
        path.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
        path.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
        path.CloseFigure();
        return path;
    }

    private void DrawRotArc(Graphics g, Point center, int radius, float degrees, Color color)
    {
        using (Pen pen = new Pen(color, 2))
        using (Pen faintPen = new Pen(Color.FromArgb(40, color), 1))
        {
            g.DrawEllipse(faintPen, center.X - radius, center.Y - radius, radius * 2, radius * 2);

            float rad = degrees * (float)Math.PI / 180f;
            int ex = center.X + (int)(radius * Math.Cos(rad - Math.PI / 2));
            int ey = center.Y + (int)(radius * Math.Sin(rad - Math.PI / 2));
            g.DrawLine(pen, center.X, center.Y, ex, ey);
        }
    }

    // =========================
    // Main
    // =========================
    [STAThread]
    public static void Main(string[] args)
    {
        int port = 3333;
        int p;
        if (args.Length == 1 && int.TryParse(args[0], out p) && p > 0)
            port = p;

        Console.WriteLine("Smart Kitchen starting on TUIO port " + port);
        Console.WriteLine("Login Marker: 25");
        Console.WriteLine("Bluetooth socket server: " + SOCKET_HOST + ":" + SOCKET_PORT);
        Console.WriteLine("Gesture socket client -> Python server: " + GESTURE_SOCKET_HOST + ":" + GESTURE_SOCKET_PORT);
        Console.WriteLine("Gaze socket client -> Python server: " + GAZE_SOCKET_HOST + ":" + GAZE_SOCKET_PORT);
        Console.WriteLine("Fallback Bluetooth file: active_bluetooth_device.txt");
        Console.WriteLine("Profile source file: SmartKitchenProfiles.txt");
        Console.WriteLine("Marker mapping:");
        Console.WriteLine("  0=Chicken  1=Tomato  2=Onion  3=Spices  4=Rice  5=Oil");
        Console.WriteLine("  6=Spoon    7=Pot     8=Knife   20=Circular Menu   25=Login");
        Console.WriteLine("Keys: C=Clear, R=Remove Last, H=Tips, M=Steps, ESC=Exit");

        System.Windows.Forms.Application.EnableVisualStyles();
        System.Windows.Forms.Application.SetCompatibleTextRenderingDefault(false);
        System.Windows.Forms.Application.Run(new SmartKitchenDemo(port));
    }
}

