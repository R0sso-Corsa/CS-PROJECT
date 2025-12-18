function change(){
    document.getElementById("testing").innerHTML = "This is a testing function.";
    window.alert("This is a testing function.");
}

function myFunction() {
    document.getElementById("demo").innerHTML = "Paragraph changed.";
}


function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash |= 0; // Convert to 32bit integer
    }
    return hash;
}

function validateLogin() {
    user = document.getElementById("username").value;
    pass = document.getElementById("password").value;
    let hashedPassword = hashString(pass);
  
    // Hardcoded credentials with hashed passwords
    const credentials = [
      { user: 'x', pass: hashString('x') },
      { user: 'user2', pass: hashString('password2') }
    ];
  
    const valid = credentials.some(cred => cred.user === user && cred.pass === hashedPassword);
  
    if (valid) {
      document.getElementById("loginForm").style.display = "none";
      document.getElementById("iframeContainer").style.display = "block";
      document.getElementById("toggleButton").style.display = "block"; // Show toggle button
      log = true; // Use global variable
    } else {
      alert("Invalid credentials");
    }
  }
  
  function toggleSlowIframe() {
    if (log) {
      const iframeContainer = document.getElementById("iframeContainer");
      iframeContainer.style.display = iframeContainer.style.display === "none" ? "block" : "none";
  } else {
    alert("You must be logged in to toggle this iframe.");
  }
}

  


function fullname() {      
    x = user
    y = pass
    return this.x + " " + this.y;

}


function ualert(){
    if (log == true) {
      alert(fullname());
    } else {
      alert("Log in...");
    }
}


let x, y, z;
x = user;
y = pass;
z = x + y;