<?php
        session_start();
    ?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COOL SITE</title>
</head>
<body>
    <H1>epic</H1>
    <nav>
        <a href="/index.php">domov </a>
        <a href="/login.php">login </a>
        <a href="logout">logout </a>
    </nav>
 
 
 
    <?php
    if(isset($_SESSION["prijavljen"]) && $_SESSION["prijavljen"] == true){
        echo"ste prijavljeni";
    }else{    

        if(isset($_POST["User"]) && isset($_POST["Pass"])){
            $user = $_POST["User"];
            $pass = $_POST["Pass"];
                if($user == "user"&& $pass == "geslo"){
                echo "Prijava uspešna!";
                $_SESSION["prijavljen"] = true;    
                }else{
                    echo "Niste prijavljeni";
                }
        }}
    ?>
 
</body>
</html>