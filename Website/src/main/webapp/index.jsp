<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<!DOCTYPE html>
<html lang="en">
	<head>
	<meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
      integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC"
      crossorigin="anonymous"
    />
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css"
    />
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/css/style.css"/>
    
    <link rel="icon" href="${pageContext.request.contextPath}/img/apple.svg" />
     <title>Apple Watch</title>
  </head>
  <body>
    <!-------Navbar----------->
    <nav class="navbar navbar-dark bg-dark">
      <div class="container">
        <h1 id="logo">
          <a href="${pageContext.request.contextPath}/index.jsp"><i class="bi bi-apple"></i></a></h1>
        <ul class="nav justify-content-center|justify-content-end">
          <li class="nav-item">
            <a class="nav-link" href="#features">About</a>
          </li>
          <li class="nav-item">
            <form action="controller" method="get">
            	<input class="nav-link" type="submit" value="Reviews"/>
            </form>
          </li>
          <li class="nav-item">
            <a class="nav-link" href="#contact">Contact</a>
          </li>
        </ul>
      </div>
    </nav>
    <!-- -----------------------Header--------------------- -->
    <header>
      <div class="home-text">
        <h2><i class="bi bi-apple"></i> WATCH</h2>
        <h1>Full Screen Ahead.</h1>
        <p>From &#8377;41900.99*</p>
        <a class="buy-btn btn btn-primary" href="https://www.amazon.in/Apple-Watch-GPS-Cellular-45mm/dp/B09HHF5H2S/ref=sr_1_2_sspa?keywords=apple%2Bwatch&qid=1650734111&sr=8-2-spons&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUFBNzFJS1Y2WUFRQUYmZW5jcnlwdGVkSWQ9QTA4NDk5NTZWSTlRWk5SMVA5SFAmZW5jcnlwdGVkQWRJZD1BMDE2NjQ3MDFJM0dSVDNIWk02U1omd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl&th=1" target="_blank" rel="noopener noreferrer"> Buy </a>
      </div>
      <div class="header-img">
        <img src="${pageContext.request.contextPath}/img/apple-watch.jpg" alt="series 7 image" />
      </div>
    </header>
    <!-- ------------------------Tile---------------------- -->
    <section id="tile-display">
      <div class="tile-img">
        <img id="img1" src="${pageContext.request.contextPath}/img/compare-apple.jpg" alt="apple watch" />
        <img id="img2" src="${pageContext.request.contextPath}/img/compare-older.jpg" alt="older watch" />
      </div>
      
      <div class="tile-text">
        <p>Our largest display yet.</p>
        <h1>Over 50% more screen area than the older version.</h1>
        <div class="tile-buttons">
          <button class="btn btn-dark" onClick="fadeImg('#img2')">Apple</button>
          <button class="btn btn-dark" onClick="showImg('#img2')">Older</button>
        </div>
      </div>
    </section>
    <!-- -------------------------Features---------------- -->
    <section id="features">
      <h2 class="feature-heading">Why Apple Watch?</h2>
      <!-- --------------Watch Display------------ -->
      <div class="watch-display">
        <div class="watch">
          <img
            src="${pageContext.request.contextPath}/img/feature-display-watch-img.jpg"
            alt="Apple Watch"
            class="feature-watch-img"
          />
          <div class="color-palette">
            <div class="watch-color aluminium-midnight"></div>
            <div class="watch-color aluminium-starlight"></div>
            <div class="watch-color aluminium-green"></div>
            <div class="watch-color aluminium-blue"></div>
            <div class="watch-color aluminium-red"></div>
            <div class="watch-color stainless-graphite"></div>
            <div class="watch-color stainless-silver"></div>
            <div class="watch-color stainless-gold"></div>
            <div class="watch-color titanium-black"></div>
            <div class="watch-color titanium-natural"></div>
          </div>
        </div>
        <!-- -------------Display Text----------- -->
        <div class="display-text">
          <h3>Apple Watch</h3>
          <p>From &#8377;41,900.99*</p>
          <a class="buy-btn btn btn-primary" href="https://www.amazon.in/Apple-Watch-GPS-Cellular-45mm/dp/B09HHF5H2S/ref=sr_1_2_sspa?keywords=apple%2Bwatch&qid=1650734111&sr=8-2-spons&spLa=ZW5jcnlwdGVkUXVhbGlmaWVyPUFBNzFJS1Y2WUFRQUYmZW5jcnlwdGVkSWQ9QTA4NDk5NTZWSTlRWk5SMVA5SFAmZW5jcnlwdGVkQWRJZD1BMDE2NjQ3MDFJM0dSVDNIWk02U1omd2lkZ2V0TmFtZT1zcF9hdGYmYWN0aW9uPWNsaWNrUmVkaXJlY3QmZG9Ob3RMb2dDbGljaz10cnVl&th=1" target="_blank" rel="noopener noreferrer"> Buy </a>
        </div>
      </div>
      <hr />
      <!-- ---------------Features Grid---------- -->
      <div class="feature-grid">
        <div class="feature size">
          <h4>45mm or 41mm</h4>
          <ul>
            <li>Swimproof<sup>1</sup></li>
            <li>Most crack-resistent front crystal</li>
            <li>Certifies IP6X dust resistent<sup>1</sup></li>
          </ul>
        </div>
        <div class="feature display">
          <img
            src="${pageContext.request.contextPath}/img/feature-display-img.png"
            alt="Always-on retina display"
          />
          <ul>
            <li>Always-on Retina display</li>
            <li>Nearly 20% larger than previous version</li>
          </ul>
        </div>
        <div class="feature oxygen">
          <img src="${pageContext.request.contextPath}/img/feature-oxygen-img.png" alt="Blood Oxygen app" />
          <p>Blood Oxygen app<sup>2</sup></p>
        </div>
        <div class="feature ecg">
          <img src="${pageContext.request.contextPath}/img/feature-ecg-img.png" alt="ECG app" />
          <p>ECG app<sup>3</sup></p>
        </div>
        <div class="feature heart">
          <img src="${pageContext.request.contextPath}/img/feature-heart-img.png" alt="Heart beat monitoring" />
          <ul>
            <li>High and low heart notification</li>
            <li>Irregular heart rhythm notification<sup>4</sup></li>
          </ul>
        </div>
        <div class="feature sos">
          <img src="${pageContext.request.contextPath}/img/feature-sos-img.png" alt="SOS" />
          <ul>
            <li>Emergency calling SOS<sup>5</sup></li>
            <li>International emergency calling<sup>6</sup></li>
            <li>Fall detection</li>
          </ul>
        </div>
      </div>
    </section>
    <!-- ----------------------------------Footer----------------------- -->
    <footer>
      <!-- --------Terms--------- -->
      <div id="terms">
        <i class="bi bi-apple"></i>
        <p>*Listed pricing is Maximum Retail Price (inclusive of all taxes).</p>
        <ol>
          <li>
            Apple Watch have a water resistance rating of 50 metres under ISO
            standard 22810:2010. This means that they may be used for
            shallow-water activities like swimming in a pool or ocean. However,
            they should not be used for scuba diving, waterskiing or other
            activities involving high-velocity water or submersion below shallow
            depth. Apple Watch is also rated IP6x dust resistant.
          </li>
          <li>
            Blood Oxygen app measurements are not intended for medical use,
            including self-diagnosis or consultation with a doctor, and are only
            designed for general fitness and wellness purposes.
          </li>
          <li>
            The ECG app requires watchOS 6 or later on Apple Watch Series 4 or
            later (not including Apple Watch SE), and iPhone 6s or later with
            the latest version of iOS. It is not intended for use by people
            under 22 years old.
          </li>
          <li>
            The ECG app requires watchOS 6 or later on Apple Watch, and iPhone
            6s or later with the latest version of iOS. It is not intended for
            use by people under 22 years old.
          </li>
          <li>
            Apple Watch (GPS + Cellular) can use a cellular connection for
            Emergency SOS. To use Emergency SOS on an Apple Watch without
            cellular, your iPhone needs to be nearby. If your iPhone isn’t
            nearby, your Apple Watch needs to be connected to a known Wi-Fi
            network and you must set up Wi-Fi Calling.
          </li>
          <li>
            The international emergency calling feature requires (GPS +
            Cellular) model
          </li>
        </ol>
      </div>
      <hr />
      <!-- ----------Contact---------- -->
      <div id="contact">
        <h3>Contact Apple</h3>
        <p>
          To shop Apple products, visit nearest Apple Store, order online or
          call 000800 040 1966.
        </p>
        <button class="btn btn-light" onclick="showMoreContact()">More &nbsp; &#8250;</button>
        <div id="office">
          <h3>Registered Office</h3>
          <p>
            19th Floor, Concorde Tower C,<br />
            UB City, No.24, Vittal Mallya Road,<br />
            Bangalore 560001, India.
            <br /><br />
            CIN: U30007KA1996PTC019630 <br /><br />
            Telephone: 91 80 4045 5150 <br />
            Fax: 91 804045 5197 <br /><br />
            bangalore_admin@apple.com
          </p>
        </div>
      </div>
      <script>
		function showMoreContact() {
		  const x = document.getElementById("office");
		  if (x.style.display === "none") {
		    x.style.display = "block";
		  } else {
		    x.style.display = "none";
		  }
		}
		
		function fadeImg(id) {
			const image = document.querySelector(id);
			var op = 1;  // initial opacity
		    var timer = setInterval(function () {
		        if (op <= 0.01){
		            clearInterval(timer);
		            image.style.display = 'none';
		        }
		        image.style.opacity = op;
		        image.style.filter = 'alpha(opacity=' + op * 100 + ")";
		        op -= op * 0.05;
		    }, 10);
		}
		
		function showImg(id) {
			const image = document.querySelector(id);
			var op = 0.1;  // initial opacity
		    image.style.display = 'block';
		    var timer = setInterval(function () {
		        if (op >= 1){
		            clearInterval(timer);
		        }
		        image.style.opacity = op;
		        image.style.filter = 'alpha(opacity=' + op * 100 + ")";
		        op += op * 0.05;
		    }, 10);
		}
	  </script>
      <div id="foot">
        <p>Copyright &#169; 2022 Apple Inc. All rights reserved.</p>
      </div>
    </footer>
  </body>
</html>
