WebCred_Plugin
==============

Plugin_UI
---------

-   Contains the front end HTML and JS part for browser plugin.
-   manifesto.js defines the requirements and permissions for the plugin

Plugin_API
----------

-   Backend Part for listening to the API requests made by plugin
-   ./Plugin_API/public/script/getScores.py should be replaced with the original score giving python script.
-   **NOTE:** the above replaced file should print only the score to terminal/STDOUT.