.. _api_TestResults:

Test Results
============

.. raw:: html

    <style type="text/css" media="screen">
    body        { font-family: verdana, arial, helvetica, sans-serif; font-size: 80%; }
    table       { font-size: 100%; }
    pre         { }
    
    /* -- heading ---------------------------------------------------------------------- */
    h1 {
        font-size: 16pt;
        color: gray;
    }
    .heading {
        margin-top: 0ex;
        margin-bottom: 1ex;
    }
    
    .heading .attribute {
        margin-top: 1ex;
        margin-bottom: 0;
    }
    
    .heading .description {
        margin-top: 4ex;
        margin-bottom: 6ex;
    }
    
    /* -- css div popup ------------------------------------------------------------------------ */
    a.popup_link {
    }
    
    a.popup_link:hover {
        color: red;
    }
    
    .popup_window {
        display: none;
        position: relative;
        left: 0px;
        top: 0px;
        /*border: solid #627173 1px; */
        padding: 10px;
        background-color: #E6E6D6;
        font-family: "Lucida Console", "Courier New", Courier, monospace;
        text-align: left;
        font-size: 8pt;
        width: 500px;
    }
    
    }
    /* -- report ------------------------------------------------------------------------ */
    #show_detail_line {
        margin-top: 3ex;
        margin-bottom: 1ex;
    }
    #result_table {
        width: 80%;
        border-collapse: collapse;
        border: 1px solid #777;
    }
    #header_row {
        font-weight: bold;
        color: white;
        background-color: #777;
    }
    #result_table td {
        border: 1px solid #777;
        padding: 2px;
    }
    #total_row  { font-weight: bold; }
    .passClass  { background-color: #6c6; }
    .failClass  { background-color: #c60; }
    .errorClass { background-color: #c00; }
    .passCase   { color: #6c6; }
    .failCase   { color: #c60; font-weight: bold; }
    .errorCase  { color: #c00; font-weight: bold; }
    .hiddenRow  { display: none; }
    .testcase   { margin-left: 2em; }
    
    
    /* -- ending ---------------------------------------------------------------------- */
    #ending {
    }
    
    </style>
    
    <script language="javascript" type="text/javascript"><!--
    output_list = Array();
    
    /* level - 0:Summary; 1:Failed; 2:All */
    function showCase(level) {
        trs = document.getElementsByTagName("tr");
        for (var i = 0; i < trs.length; i++) {
            tr = trs[i];
            id = tr.id;
            if (id.substr(0,2) == 'ft') {
                if (level < 1) {
                    tr.className = 'hiddenRow';
                }
                else {
                    tr.className = '';
                }
            }
            if (id.substr(0,2) == 'pt') {
                if (level > 1) {
                    tr.className = '';
                }
                else {
                    tr.className = 'hiddenRow';
                }
            }
        }
    }
    
    
    function showClassDetail(cid, count) {
        var id_list = Array(count);
        var toHide = 1;
        for (var i = 0; i < count; i++) {
            tid0 = 't' + cid.substr(1) + '.' + (i+1);
            tid = 'f' + tid0;
            tr = document.getElementById(tid);
            if (!tr) {
                tid = 'p' + tid0;
                tr = document.getElementById(tid);
            }
            id_list[i] = tid;
            if (tr.className) {
                toHide = 0;
            }
        }
        for (var i = 0; i < count; i++) {
            tid = id_list[i];
            if (toHide) {
                document.getElementById('div_'+tid).style.display = 'none'
                document.getElementById(tid).className = 'hiddenRow';
            }
            else {
                document.getElementById(tid).className = '';
            }
        }
    }
    
    
    function showTestDetail(div_id){
        var details_div = document.getElementById(div_id)
        var displayState = details_div.style.display
        // alert(displayState)
        if (displayState != 'block' ) {
            displayState = 'block'
            details_div.style.display = 'block'
        }
        else {
            details_div.style.display = 'none'
        }
    }
    
    
    function html_escape(s) {
        s = s.replace(/&/g,'&amp;');
        s = s.replace(/</g,'&lt;');
        s = s.replace(/>/g,'&gt;');
        return s;
    }
    
    /* obsoleted by detail in <div>
    function showOutput(id, name) {
        var w = window.open("", //url
                        name,
                        "resizable,scrollbars,status,width=800,height=450");
        d = w.document;
        d.write("<pre>");
        d.write(html_escape(output_list[id]));
        d.write("\n");
        d.write("<a href='javascript:window.close()'>close</a>\n");
        d.write("</pre>\n");
        d.close();
    }
    */
    --></script>
    
    <div class='heading'>
    <p class='attribute'><strong>Start Time:</strong> 2013-11-12 13:34:53</p>
    <p class='attribute'><strong>Duration:</strong> 0:00:31.732498</p>
    <p class='attribute'><strong>Status:</strong> Pass 117</p>
    
    <p class='description'>SimPEG Test Report was automatically generated.</p>
    </div>
    
    
    
    <p id='show_detail_line'>Show
    <a href='javascript:showCase(0)'>Summary</a>
    <a href='javascript:showCase(1)'>Failed</a>
    <a href='javascript:showCase(2)'>All</a>
    </p>
    <table id='result_table'>
    <colgroup>
    <col align='left' />
    <col align='right' />
    <col align='right' />
    <col align='right' />
    <col align='right' />
    <col align='right' />
    </colgroup>
    <tr id='header_row'>
        <td>Test Group/Test case</td>
        <td>Count</td>
        <td>Pass</td>
        <td>Fail</td>
        <td>Error</td>
        <td>View</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_basemesh.TestBaseMesh</td>
        <td>11</td>
        <td>11</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c1',11)">Detail</a></td>
    </tr>
    
    <tr id='pt1.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_meshDimensions</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_nc</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_nc_xyz</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_ne</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_nf</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_numbers</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_CC_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_E_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.9' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_E_V</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.10' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_F_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt1.11' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_F_V</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_basemesh.TestMeshNumbers2D</td>
        <td>11</td>
        <td>11</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c2',11)">Detail</a></td>
    </tr>
    
    <tr id='pt2.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_meshDimensions</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_nc</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_nc_xyz</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_ne</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_nf</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_numbers</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_CC_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_E_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.9' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_E_V</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.10' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_F_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt2.11' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mesh_r_F_V</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_forward_DCproblem.DCProblemTests</td>
        <td>4</td>
        <td>4</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c3',4)">Detail</a></td>
    </tr>
    
    <tr id='pt3.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_adjoint</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt3.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_dataObj</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt3.2')" >
            pass</a>
    
        <div id='div_pt3.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt3.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt3.2: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	5.517e+01		4.463e+01		nan
    1	1.00e-02	1.516e+00		4.609e-01		1.986
    2	1.00e-03	1.101e-01		4.624e-03		1.999
    3	1.00e-04	1.059e-02		4.626e-05		2.000
    4	1.00e-05	1.055e-03		4.626e-07		2.000
    5	1.00e-06	1.055e-04		4.623e-09		2.000
    6	1.00e-07	1.055e-05		4.504e-11		2.011
    ========================= PASS! =========================
    Awesome, Rowan, just awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt3.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_misfit</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt3.3')" >
            pass</a>
    
        <div id='div_pt3.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt3.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt3.3: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.180e-01		6.488e-03		nan
    1	1.00e-02	1.175e-02		6.401e-05		2.006
    2	1.00e-03	1.175e-03		6.390e-07		2.001
    3	1.00e-04	1.175e-04		6.389e-09		2.000
    4	1.00e-05	1.175e-05		6.389e-11		2.000
    5	1.00e-06	1.175e-06		6.392e-13		2.000
    6	1.00e-07	1.175e-07		7.710e-15		1.919
    ========================= PASS! =========================
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt3.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_modelObj</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt3.4')" >
            pass</a>
    
        <div id='div_pt3.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt3.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt3.4: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	6.898e+00		7.074e+00		nan
    1	1.00e-02	5.316e-02		7.074e-02		2.000
    2	1.00e-03	1.051e-03		7.074e-04		2.000
    3	1.00e-04	1.688e-04		7.074e-06		2.000
    4	1.00e-05	1.752e-05		7.074e-08		2.000
    5	1.00e-06	1.758e-06		7.074e-10		2.000
    6	1.00e-07	1.759e-07		7.076e-12		2.000
    ========================= PASS! =========================
    Testing is important.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_forward_problem.ProblemTests</td>
        <td>2</td>
        <td>2</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c4',2)">Detail</a></td>
    </tr>
    
    <tr id='pt4.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_modelTransform</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt4.1')" >
            pass</a>
    
        <div id='div_pt4.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt4.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt4.1: SimPEG.forward.Problem: Testing Model Transform
    ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.762e-01		4.105e-17		nan
    1	1.00e-02	2.762e-02		2.532e-17		0.210
    2	1.00e-03	2.762e-03		4.066e-17		-0.206
    3	1.00e-04	2.762e-04		4.297e-17		-0.024
    4	1.00e-05	2.762e-05		5.494e-17		-0.107
    5	1.00e-06	2.762e-06		4.696e-17		0.068
    6	1.00e-07	2.762e-07		2.834e-17		0.219
    ========================= PASS! =========================
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt4.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_regularization</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt4.2')" >
            pass</a>
    
        <div id='div_pt4.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt4.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt4.2: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	5.835e-01		2.875e-01		nan
    1	1.00e-02	3.247e-02		2.875e-03		2.000
    2	1.00e-03	2.989e-03		2.875e-05		2.000
    3	1.00e-04	2.963e-04		2.875e-07		2.000
    4	1.00e-05	2.960e-05		2.875e-09		2.000
    5	1.00e-06	2.960e-06		2.875e-11		2.000
    6	1.00e-07	2.960e-07		2.879e-13		1.999
    ========================= PASS! =========================
    Awesome, Rowan, just awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_interpolation.TestInterpolation1D</td>
        <td>2</td>
        <td>2</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c5',2)">Detail</a></td>
    </tr>
    
    <tr id='pt5.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderCC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt5.1')" >
            pass</a>
    
        <div id='div_pt5.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt5.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt5.1: 
    uniformTensorMesh:  Interpolation 1D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.14e-01   |
      16  |  6.31e-02   |   3.3843    |  1.7589
      32  |  1.72e-02   |   3.6712    |  1.8763
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation 1D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  3.35e-01   |
      16  |  1.18e-01   |   2.8469    |  1.8016
      32  |  2.27e-02   |   5.1806    |  3.9741
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt5.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt5.2')" >
            pass</a>
    
        <div id='div_pt5.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt5.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt5.2: 
    uniformTensorMesh:  Interpolation 1D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.98e-01   |
      16  |  6.69e-02   |   4.4487    |  2.1534
      32  |  1.55e-02   |   4.3044    |  2.1058
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation 1D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.16e-01   |
      16  |  1.60e-01   |   4.4731    |  1.8851
      32  |  3.72e-02   |   4.3056    |  2.5981
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_interpolation.TestInterpolation2d</td>
        <td>6</td>
        <td>6</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c6',6)">Detail</a></td>
    </tr>
    
    <tr id='pt6.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderCC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt6.1')" >
            pass</a>
    
        <div id='div_pt6.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt6.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt6.1: 
    uniformTensorMesh:  Interpolation 2D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.48e-02   |
      16  |  1.79e-02   |   4.1840    |  2.0649
      32  |  4.38e-03   |   4.0830    |  2.0296
      64  |  1.16e-03   |   3.7604    |  1.9109
    ---------------------------------------------
    Testing is important.
    
    
    randomTensorMesh:  Interpolation 2D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.76e-02   |
      16  |  6.08e-02   |   0.7830    |  -1.0896
      32  |  1.14e-02   |   5.3162    |  1.7097
      64  |  1.19e-03   |   9.6340    |  3.5674
    ---------------------------------------------
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt6.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderEx</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt6.2')" >
            pass</a>
    
        <div id='div_pt6.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt6.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt6.2: 
    uniformTensorMesh:  Interpolation 2D: Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  6.98e-02   |
      16  |  1.85e-02   |   3.7771    |  1.9173
      32  |  4.76e-03   |   3.8827    |  1.9571
      64  |  1.20e-03   |   3.9676    |  1.9883
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 2D: Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.61e-01   |
      16  |  3.89e-02   |   6.7177    |  2.4144
      32  |  1.41e-02   |   2.7587    |  1.3921
      64  |  4.59e-03   |   3.0696    |  1.9255
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt6.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderEy</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt6.3')" >
            pass</a>
    
        <div id='div_pt6.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt6.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt6.3: 
    uniformTensorMesh:  Interpolation 2D: Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  6.97e-02   |
      16  |  1.87e-02   |   3.7164    |  1.8939
      32  |  4.74e-03   |   3.9516    |  1.9825
      64  |  1.14e-03   |   4.1658    |  2.0586
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 2D: Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.82e-01   |
      16  |  3.85e-02   |   4.7305    |  2.2574
      32  |  9.33e-03   |   4.1268    |  2.3056
      64  |  3.29e-03   |   2.8348    |  1.1080
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt6.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderFx</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt6.4')" >
            pass</a>
    
        <div id='div_pt6.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt6.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt6.4: 
    uniformTensorMesh:  Interpolation 2D: Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.48e-02   |
      16  |  1.79e-02   |   4.1840    |  2.0649
      32  |  4.38e-03   |   4.0830    |  2.0296
      64  |  1.16e-03   |   3.7604    |  1.9109
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation 2D: Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.78e-02   |
      16  |  4.87e-02   |   2.0098    |  1.1486
      32  |  6.39e-03   |   7.6229    |  2.5269
      64  |  3.06e-03   |   2.0873    |  1.0490
    ---------------------------------------------
    You are awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt6.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderFy</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt6.5')" >
            pass</a>
    
        <div id='div_pt6.5' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt6.5').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt6.5: 
    uniformTensorMesh:  Interpolation 2D: Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.59e-02   |
      16  |  1.90e-02   |   3.9914    |  1.9969
      32  |  4.62e-03   |   4.1142    |  2.0406
      64  |  1.15e-03   |   4.0248    |  2.0089
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation 2D: Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.34e-01   |
      16  |  2.73e-02   |   4.9171    |  2.4681
      32  |  1.10e-02   |   2.4900    |  1.2825
      64  |  3.06e-03   |   3.5774    |  1.9074
    ---------------------------------------------
    Testing is important.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt6.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt6.6')" >
            pass</a>
    
        <div id='div_pt6.6' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt6.6').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt6.6: 
    uniformTensorMesh:  Interpolation 2D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  6.98e-02   |
      16  |  1.85e-02   |   3.7771    |  1.9173
      32  |  4.76e-03   |   3.8827    |  1.9571
      64  |  1.20e-03   |   3.9676    |  1.9883
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    randomTensorMesh:  Interpolation 2D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.05e-01   |
      16  |  4.37e-02   |   4.6889    |  2.7894
      32  |  1.30e-02   |   3.3576    |  1.7086
      64  |  3.69e-03   |   3.5290    |  1.8138
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_interpolation.TestInterpolation3D</td>
        <td>8</td>
        <td>8</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c7',8)">Detail</a></td>
    </tr>
    
    <tr id='pt7.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderCC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.1')" >
            pass</a>
    
        <div id='div_pt7.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.1: 
    uniformTensorMesh:  Interpolation CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.56e-02   |
      16  |  1.87e-02   |   4.0396    |  2.0142
      32  |  4.52e-03   |   4.1409    |  2.0499
      64  |  1.19e-03   |   3.8144    |  1.9315
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    randomTensorMesh:  Interpolation CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.03e-01   |
      16  |  5.43e-02   |   1.8919    |  1.5786
      32  |  1.33e-02   |   4.0699    |  1.4626
      64  |  3.40e-03   |   3.9265    |  2.7539
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderEx</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.2')" >
            pass</a>
    
        <div id='div_pt7.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.2: 
    uniformTensorMesh:  Interpolation Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.04e-02   |
      16  |  1.86e-02   |   3.7784    |  1.9178
      32  |  4.78e-03   |   3.8981    |  1.9628
      64  |  1.20e-03   |   3.9741    |  1.9906
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.52e-01   |
      16  |  2.58e-02   |   9.7516    |  5.3251
      32  |  1.69e-02   |   1.5285    |  0.4736
      64  |  3.83e-03   |   4.4134    |  2.6664
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderEy</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.3')" >
            pass</a>
    
        <div id='div_pt7.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.3: 
    uniformTensorMesh:  Interpolation Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.03e-02   |
      16  |  1.83e-02   |   3.8430    |  1.9422
      32  |  4.72e-03   |   3.8783    |  1.9554
      64  |  1.17e-03   |   4.0244    |  2.0088
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.70e-02   |
      16  |  2.92e-02   |   2.6359    |  2.0836
      32  |  1.66e-02   |   1.7612    |  0.7062
      64  |  3.79e-03   |   4.3764    |  2.5121
    ---------------------------------------------
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderEz</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.4')" >
            pass</a>
    
        <div id='div_pt7.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.4: 
    uniformTensorMesh:  Interpolation Ez
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.04e-02   |
      16  |  1.87e-02   |   3.7587    |  1.9102
      32  |  4.79e-03   |   3.9106    |  1.9674
      64  |  1.17e-03   |   4.1061    |  2.0378
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation Ez
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.56e-01   |
      16  |  3.70e-02   |   4.2142    |  1.7123
      32  |  1.16e-02   |   3.1865    |  1.2409
      64  |  4.18e-03   |   2.7767    |  1.8503
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderFx</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.5')" >
            pass</a>
    
        <div id='div_pt7.5' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.5').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.5: 
    uniformTensorMesh:  Interpolation Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.56e-02   |
      16  |  1.87e-02   |   4.0396    |  2.0142
      32  |  4.52e-03   |   4.1409    |  2.0499
      64  |  1.19e-03   |   3.8144    |  1.9315
    ---------------------------------------------
    The test be workin!
    
    
    randomTensorMesh:  Interpolation Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.85e-02   |
      16  |  3.26e-02   |   1.4891    |  0.4457
      32  |  7.59e-03   |   4.2884    |  2.0210
      64  |  2.21e-03   |   3.4371    |  2.3960
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderFy</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.6')" >
            pass</a>
    
        <div id='div_pt7.6' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.6').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.6: 
    uniformTensorMesh:  Interpolation Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.55e-02   |
      16  |  1.86e-02   |   4.0668    |  2.0239
      32  |  4.25e-03   |   4.3625    |  2.1252
      64  |  1.12e-03   |   3.7855    |  1.9205
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.26e-01   |
      16  |  5.43e-02   |   2.3127    |  1.1973
      32  |  6.95e-03   |   7.8090    |  3.7839
      64  |  2.07e-03   |   3.3670    |  1.4675
    ---------------------------------------------
    You get a gold star!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderFz</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.7')" >
            pass</a>
    
        <div id='div_pt7.7' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.7').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.7: 
    uniformTensorMesh:  Interpolation Fz
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.56e-02   |
      16  |  1.87e-02   |   4.0457    |  2.0164
      32  |  4.58e-03   |   4.0787    |  2.0281
      64  |  1.19e-03   |   3.8390    |  1.9407
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation Fz
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.17e-01   |
      16  |  2.69e-02   |   4.3374    |  2.1585
      32  |  1.64e-02   |   1.6402    |  0.8257
      64  |  3.86e-03   |   4.2449    |  1.7674
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt7.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt7.8')" >
            pass</a>
    
        <div id='div_pt7.8' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt7.8').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt7.8: 
    uniformTensorMesh:  Interpolation N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.04e-02   |
      16  |  1.86e-02   |   3.7784    |  1.9178
      32  |  4.78e-03   |   3.8981    |  1.9628
      64  |  1.20e-03   |   3.9741    |  1.9906
    ---------------------------------------------
    The test be workin!
    
    
    randomTensorMesh:  Interpolation N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.78e-01   |
      16  |  3.66e-02   |   4.8776    |  3.1782
      32  |  1.61e-02   |   2.2707    |  1.1549
      64  |  3.17e-03   |   5.0740    |  2.0894
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_LogicallyOrthogonalMesh.BasicLOMTests</td>
        <td>8</td>
        <td>8</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c8',8)">Detail</a></td>
    </tr>
    
    <tr id='pt8.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_area_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_grid</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_normals</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_tangents</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt8.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_massMatrices.TestInnerProducts: Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts.</td>
        <td>6</td>
        <td>6</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c9',6)">Detail</a></td>
    </tr>
    
    <tr id='pt9.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order1_edges</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt9.1')" >
            pass</a>
    
        <div id='div_pt9.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt9.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt9.1: 
    uniformTensorMesh:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.42e-03   |
      32  |  6.06e-04   |   4.0001    |  2.0000
    ---------------------------------------------
    Happy little convergence test!
    
    
    uniformLOM:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.42e-03   |
      32  |  6.06e-04   |   4.0001    |  2.0000
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.81e-03   |
      32  |  7.11e-04   |   3.9432    |  1.9794
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt9.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order1_faces</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt9.2')" >
            pass</a>
    
        <div id='div_pt9.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt9.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt9.2: 
    uniformTensorMesh:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.29e-04   |
      32  |  1.57e-04   |   3.9978    |  1.9992
    ---------------------------------------------
    Testing is important.
    
    
    uniformLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.29e-04   |
      32  |  1.57e-04   |   3.9978    |  1.9992
    ---------------------------------------------
    And then everyone was happy.
    
    
    rotateLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.08e-04   |
      32  |  7.07e-05   |   4.3564    |  2.1231
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt9.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order3_edges</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt9.3')" >
            pass</a>
    
        <div id='div_pt9.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt9.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt9.3: 
    uniformTensorMesh:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.99e-03   |
      32  |  1.75e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    That was easy!
    
    
    uniformLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.99e-03   |
      32  |  1.75e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    You get a gold star!
    
    
    rotateLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  7.70e-03   |
      32  |  1.94e-03   |   3.9622    |  1.9863
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt9.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order3_faces</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt9.4')" >
            pass</a>
    
        <div id='div_pt9.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt9.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt9.4: 
    uniformTensorMesh:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.68e-03   |
      32  |  6.69e-04   |   3.9982    |  1.9993
    ---------------------------------------------
    And then everyone was happy.
    
    
    uniformLOM:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.68e-03   |
      32  |  6.69e-04   |   3.9982    |  1.9993
    ---------------------------------------------
    Yay passed!
    
    
    rotateLOM:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.15e-03   |
      32  |  5.25e-04   |   4.0845    |  2.0302
    ---------------------------------------------
    Yay passed!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt9.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order6_edges</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt9.5')" >
            pass</a>
    
        <div id='div_pt9.5' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt9.5').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt9.5: 
    uniformTensorMesh:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.79e-03   |
      32  |  1.70e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    The test be workin!
    
    
    uniformLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.79e-03   |
      32  |  1.70e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    And then everyone was happy.
    
    
    rotateLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  7.49e-03   |
      32  |  1.89e-03   |   3.9617    |  1.9861
    ---------------------------------------------
    The test be workin!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt9.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order6_faces</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt9.6')" >
            pass</a>
    
        <div id='div_pt9.6' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt9.6').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt9.6: 
    uniformTensorMesh:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.10e-03   |
      32  |  7.74e-04   |   3.9981    |  1.9993
    ---------------------------------------------
    You get a gold star!
    
    
    uniformLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.10e-03   |
      32  |  7.74e-04   |   3.9981    |  1.9993
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.54e-03   |
      32  |  6.23e-04   |   4.0741    |  2.0265
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_massMatrices.TestInnerProducts2D: Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts.</td>
        <td>6</td>
        <td>6</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c10',6)">Detail</a></td>
    </tr>
    
    <tr id='pt10.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order1_edges</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt10.1')" >
            pass</a>
    
        <div id='div_pt10.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt10.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt10.1: 
    uniformTensorMesh:  2D Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  5.60e+00   |
       8  |  1.40e+00   |   4.0040    |  2.0014
      16  |  3.50e-01   |   4.0010    |  2.0004
      32  |  8.74e-02   |   4.0002    |  2.0001
      64  |  2.18e-02   |   4.0001    |  2.0000
     128  |  5.46e-03   |   4.0000    |  2.0000
    ---------------------------------------------
    The test be workin!
    
    
    uniformLOM:  2D Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  5.60e+00   |
       8  |  1.40e+00   |   4.0040    |  2.0014
      16  |  3.50e-01   |   4.0010    |  2.0004
      32  |  8.74e-02   |   4.0002    |  2.0001
      64  |  2.18e-02   |   4.0001    |  2.0000
     128  |  5.46e-03   |   4.0000    |  2.0000
    ---------------------------------------------
    You get a gold star!
    
    
    rotateLOM:  2D Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  5.38e+00   |
       8  |  1.31e+00   |   4.1168    |  2.0415
      16  |  3.23e-01   |   4.0449    |  2.0161
      32  |  8.03e-02   |   4.0235    |  2.0085
      64  |  2.00e-02   |   4.0155    |  2.0056
     128  |  5.00e-03   |   4.0038    |  2.0014
    ---------------------------------------------
    You are awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt10.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order1_faces</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt10.2')" >
            pass</a>
    
        <div id='div_pt10.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt10.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt10.2: 
    uniformTensorMesh:  2D Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.46e+00   |
       8  |  1.62e+00   |   3.9970    |  1.9989
      16  |  4.04e-01   |   3.9992    |  1.9997
      32  |  1.01e-01   |   3.9998    |  1.9999
      64  |  2.53e-02   |   4.0000    |  2.0000
     128  |  6.32e-03   |   4.0000    |  2.0000
    ---------------------------------------------
    Happy little convergence test!
    
    
    uniformLOM:  2D Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.46e+00   |
       8  |  1.62e+00   |   3.9970    |  1.9989
      16  |  4.04e-01   |   3.9992    |  1.9997
      32  |  1.01e-01   |   3.9998    |  1.9999
      64  |  2.53e-02   |   4.0000    |  2.0000
     128  |  6.32e-03   |   4.0000    |  2.0000
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    rotateLOM:  2D Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.01e+00   |
       8  |  1.49e+00   |   4.0334    |  2.0120
      16  |  3.69e-01   |   4.0329    |  2.0118
      32  |  9.22e-02   |   4.0052    |  2.0019
      64  |  2.30e-02   |   4.0132    |  2.0048
     128  |  5.74e-03   |   4.0009    |  2.0003
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt10.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order2_faces</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt10.3')" >
            pass</a>
    
        <div id='div_pt10.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt10.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt10.3: 
    uniformTensorMesh:  2D Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.32e+01   |
       8  |  8.29e+00   |   4.0000    |  2.0000
      16  |  2.07e+00   |   4.0000    |  2.0000
      32  |  5.18e-01   |   4.0000    |  2.0000
      64  |  1.30e-01   |   4.0000    |  2.0000
     128  |  3.24e-02   |   4.0000    |  2.0000
    ---------------------------------------------
    That was easy!
    
    
    uniformLOM:  2D Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.32e+01   |
       8  |  8.29e+00   |   4.0000    |  2.0000
      16  |  2.07e+00   |   4.0000    |  2.0000
      32  |  5.18e-01   |   4.0000    |  2.0000
      64  |  1.30e-01   |   4.0000    |  2.0000
     128  |  3.24e-02   |   4.0000    |  2.0000
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  2D Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.24e+01   |
       8  |  8.14e+00   |   3.9797    |  1.9927
      16  |  2.04e+00   |   3.9944    |  1.9980
      32  |  5.11e-01   |   3.9923    |  1.9972
      64  |  1.28e-01   |   4.0007    |  2.0003
     128  |  3.19e-02   |   3.9975    |  1.9991
    ---------------------------------------------
    You are awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt10.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order3_edges</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt10.4')" >
            pass</a>
    
        <div id='div_pt10.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt10.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt10.4: 
    uniformTensorMesh:  2D Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  9.36e+00   |
       8  |  2.34e+00   |   4.0002    |  2.0001
      16  |  5.85e-01   |   4.0001    |  2.0000
      32  |  1.46e-01   |   4.0000    |  2.0000
      64  |  3.66e-02   |   4.0000    |  2.0000
     128  |  9.14e-03   |   4.0000    |  2.0000
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    uniformLOM:  2D Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  9.36e+00   |
       8  |  2.34e+00   |   4.0002    |  2.0001
      16  |  5.85e-01   |   4.0001    |  2.0000
      32  |  1.46e-01   |   4.0000    |  2.0000
      64  |  3.66e-02   |   4.0000    |  2.0000
     128  |  9.14e-03   |   4.0000    |  2.0000
    ---------------------------------------------
    Well done Rowan!
    
    
    rotateLOM:  2D Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.11e+00   |
       8  |  1.58e+00   |   4.5124    |  2.1739
      16  |  3.74e-01   |   4.2134    |  2.0750
      32  |  9.08e-02   |   4.1184    |  2.0421
      64  |  2.23e-02   |   4.0739    |  2.0264
     128  |  5.54e-03   |   4.0255    |  2.0092
    ---------------------------------------------
    Testing is important.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt10.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order3_faces</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt10.5')" >
            pass</a>
    
        <div id='div_pt10.5' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt10.5').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt10.5: 
    uniformTensorMesh:  2D Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.89e+01   |
       8  |  9.73e+00   |   4.0002    |  2.0001
      16  |  2.43e+00   |   4.0001    |  2.0000
      32  |  6.08e-01   |   4.0000    |  2.0000
      64  |  1.52e-01   |   4.0000    |  2.0000
     128  |  3.80e-02   |   4.0000    |  2.0000
    ---------------------------------------------
    Testing is important.
    
    
    uniformLOM:  2D Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.89e+01   |
       8  |  9.73e+00   |   4.0002    |  2.0001
      16  |  2.43e+00   |   4.0001    |  2.0000
      32  |  6.08e-01   |   4.0000    |  2.0000
      64  |  1.52e-01   |   4.0000    |  2.0000
     128  |  3.80e-02   |   4.0000    |  2.0000
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    rotateLOM:  2D Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.72e+01   |
       8  |  9.31e+00   |   3.9994    |  1.9998
      16  |  2.32e+00   |   4.0134    |  2.0048
      32  |  5.81e-01   |   3.9964    |  1.9987
      64  |  1.45e-01   |   4.0074    |  2.0027
     128  |  3.62e-02   |   3.9984    |  1.9994
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt10.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order6_edges</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt10.6')" >
            pass</a>
    
        <div id='div_pt10.6' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt10.6').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt10.6: 
    uniformTensorMesh:  2D Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  9.70e-01   |
       8  |  2.42e-01   |   4.0063    |  2.0023
      16  |  6.05e-02   |   4.0016    |  2.0006
      32  |  1.51e-02   |   4.0004    |  2.0001
      64  |  3.78e-03   |   4.0001    |  2.0000
     128  |  9.46e-04   |   4.0000    |  2.0000
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    uniformLOM:  2D Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  9.70e-01   |
       8  |  2.42e-01   |   4.0063    |  2.0023
      16  |  6.05e-02   |   4.0016    |  2.0006
      32  |  1.51e-02   |   4.0004    |  2.0001
      64  |  3.78e-03   |   4.0001    |  2.0000
     128  |  9.46e-04   |   4.0000    |  2.0000
    ---------------------------------------------
    Testing is important.
    
    
    rotateLOM:  2D Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.05e+00   |
       8  |  1.04e+00   |   2.9506    |  1.5610
      16  |  2.89e-01   |   3.5845    |  1.8418
      32  |  7.67e-02   |   3.7658    |  1.9129
      64  |  1.98e-02   |   3.8708    |  1.9526
     128  |  5.03e-03   |   3.9418    |  1.9789
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestAveraging2D</td>
        <td>5</td>
        <td>5</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c11',5)">Detail</a></td>
    </tr>
    
    <tr id='pt11.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderE2CC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt11.1')" >
            pass</a>
    
        <div id='div_pt11.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt11.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt11.1: 
    uniformTensorMesh:  Averaging 2D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    Yay passed!
    
    
    uniformLOM:  Averaging 2D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    That was easy!
    
    
    rotateLOM:  Averaging 2D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.88e-03   |
       8  |  1.82e-03   |   3.7846    |  1.9202
      16  |  4.66e-04   |   3.8970    |  1.9624
      32  |  1.18e-04   |   3.9552    |  1.9837
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderF2CC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt11.2')" >
            pass</a>
    
        <div id='div_pt11.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt11.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt11.2: 
    uniformTensorMesh:  Averaging 2D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Averaging 2D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Averaging 2D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.88e-03   |
       8  |  1.82e-03   |   3.7846    |  1.9202
      16  |  4.66e-04   |   3.8970    |  1.9624
      32  |  1.18e-04   |   3.9552    |  1.9837
    ---------------------------------------------
    The test be workin!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2CC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt11.3')" >
            pass</a>
    
        <div id='div_pt11.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt11.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt11.3: 
    uniformTensorMesh:  Averaging 2D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.37e-02   |
       8  |  3.52e-03   |   3.8978    |  1.9627
      16  |  8.90e-04   |   3.9561    |  1.9841
      32  |  2.24e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    Well done Rowan!
    
    
    uniformLOM:  Averaging 2D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.37e-02   |
       8  |  3.52e-03   |   3.8978    |  1.9627
      16  |  8.90e-04   |   3.9561    |  1.9841
      32  |  2.24e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    That was easy!
    
    
    rotateLOM:  Averaging 2D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.38e-02   |
       8  |  3.64e-03   |   3.7900    |  1.9222
      16  |  9.33e-04   |   3.8980    |  1.9627
      32  |  2.36e-04   |   3.9577    |  1.9847
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2E</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt11.4')" >
            pass</a>
    
        <div id='div_pt11.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt11.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt11.4: 
    uniformTensorMesh:  Averaging 2D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    Happy little convergence test!
    
    
    uniformLOM:  Averaging 2D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    That was easy!
    
    
    rotateLOM:  Averaging 2D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.02e-02   |
       8  |  2.73e-03   |   3.7288    |  1.8987
      16  |  7.34e-04   |   3.7229    |  1.8964
      32  |  1.86e-04   |   3.9505    |  1.9820
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2F</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt11.5')" >
            pass</a>
    
        <div id='div_pt11.5' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt11.5').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt11.5: 
    uniformTensorMesh:  Averaging 2D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    Testing is important.
    
    
    uniformLOM:  Averaging 2D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    And then everyone was happy.
    
    
    rotateLOM:  Averaging 2D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.02e-02   |
       8  |  2.73e-03   |   3.7288    |  1.8987
      16  |  7.34e-04   |   3.7229    |  1.8964
      32  |  1.86e-04   |   3.9505    |  1.9820
    ---------------------------------------------
    Yay passed!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestAveraging3D</td>
        <td>5</td>
        <td>5</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c12',5)">Detail</a></td>
    </tr>
    
    <tr id='pt12.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderE2CC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt12.1')" >
            pass</a>
    
        <div id='div_pt12.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt12.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt12.1: 
    uniformTensorMesh:  Averaging 3D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  8.53e-03   |
       8  |  2.47e-03   |   3.4463    |  1.7851
      16  |  6.63e-04   |   3.7311    |  1.8996
      32  |  1.71e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Go Test Go!
    
    
    uniformLOM:  Averaging 3D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  8.53e-03   |
       8  |  2.47e-03   |   3.4463    |  1.7851
      16  |  6.63e-04   |   3.7311    |  1.8996
      32  |  1.71e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Averaging 3D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  9.08e-03   |
       8  |  2.55e-03   |   3.5588    |  1.8314
      16  |  7.77e-04   |   3.2852    |  1.7160
      32  |  2.08e-04   |   3.7415    |  1.9036
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderF2CC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt12.2')" >
            pass</a>
    
        <div id='div_pt12.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt12.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt12.2: 
    uniformTensorMesh:  Averaging 3D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.26e-03   |
       8  |  1.24e-03   |   3.4463    |  1.7851
      16  |  3.32e-04   |   3.7311    |  1.8996
      32  |  8.57e-05   |   3.8674    |  1.9514
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Averaging 3D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.26e-03   |
       8  |  1.24e-03   |   3.4463    |  1.7851
      16  |  3.32e-04   |   3.7311    |  1.8996
      32  |  8.57e-05   |   3.8674    |  1.9514
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    rotateLOM:  Averaging 3D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.54e-03   |
       8  |  1.28e-03   |   3.5589    |  1.8314
      16  |  3.88e-04   |   3.2842    |  1.7155
      32  |  1.04e-04   |   3.7417    |  1.9037
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2CC</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt12.3')" >
            pass</a>
    
        <div id='div_pt12.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt12.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt12.3: 
    uniformTensorMesh:  Averaging 3D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.28e-02   |
       8  |  3.71e-03   |   3.4463    |  1.7851
      16  |  9.95e-04   |   3.7311    |  1.8996
      32  |  2.57e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Happy little convergence test!
    
    
    uniformLOM:  Averaging 3D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.28e-02   |
       8  |  3.71e-03   |   3.4463    |  1.7851
      16  |  9.95e-04   |   3.7311    |  1.8996
      32  |  2.57e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Averaging 3D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.36e-02   |
       8  |  3.83e-03   |   3.5589    |  1.8314
      16  |  1.17e-03   |   3.2862    |  1.7164
      32  |  3.11e-04   |   3.7413    |  1.9036
    ---------------------------------------------
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2E</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt12.4')" >
            pass</a>
    
        <div id='div_pt12.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt12.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt12.4: 
    uniformTensorMesh:  Averaging 3D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.88e-02   |
       8  |  4.99e-03   |   3.7613    |  1.9112
      16  |  1.29e-03   |   3.8779    |  1.9553
      32  |  3.27e-04   |   3.9382    |  1.9775
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Averaging 3D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.88e-02   |
       8  |  4.99e-03   |   3.7613    |  1.9112
      16  |  1.29e-03   |   3.8779    |  1.9553
      32  |  3.27e-04   |   3.9382    |  1.9775
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    rotateLOM:  Averaging 3D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.41e-02   |
       8  |  6.11e-03   |   3.9523    |  1.9827
      16  |  1.70e-03   |   3.5908    |  1.8443
      32  |  4.41e-04   |   3.8566    |  1.9473
    ---------------------------------------------
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2F</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt12.5')" >
            pass</a>
    
        <div id='div_pt12.5' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt12.5').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt12.5: 
    uniformTensorMesh:  Averaging 3D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.78e-02   |
       8  |  4.87e-03   |   3.6557    |  1.8701
      16  |  1.27e-03   |   3.8285    |  1.9368
      32  |  3.25e-04   |   3.9144    |  1.9688
    ---------------------------------------------
    Well done Rowan!
    
    
    uniformLOM:  Averaging 3D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.78e-02   |
       8  |  4.87e-03   |   3.6557    |  1.8701
      16  |  1.27e-03   |   3.8285    |  1.9368
      32  |  3.25e-04   |   3.9144    |  1.9688
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Averaging 3D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.01e-02   |
       8  |  5.39e-03   |   3.7357    |  1.9014
      16  |  1.49e-03   |   3.6148    |  1.8539
      32  |  3.92e-04   |   3.7997    |  1.9259
    ---------------------------------------------
    Yay passed!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestCurl</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c13',1)">Detail</a></td>
    </tr>
    
    <tr id='pt13.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt13.1')" >
            pass</a>
    
        <div id='div_pt13.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt13.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt13.1: 
    uniformTensorMesh:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    You are awesome.
    
    
    uniformLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.71e-01   |
       8  |  8.86e-02   |   5.3170    |  2.4106
      16  |  1.70e-02   |   5.2040    |  2.3796
      32  |  3.77e-03   |   4.5126    |  2.1740
    ---------------------------------------------
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestFaceDiv</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c14',1)">Detail</a></td>
    </tr>
    
    <tr id='pt14.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt14.1')" >
            pass</a>
    
        <div id='div_pt14.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt14.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt14.1: 
    uniformTensorMesh:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    uniformLOM:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.08e-03   |
      16  |  9.53e-04   |   9.5374    |  3.2536
      32  |  2.75e-04   |   3.4594    |  1.7905
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestFaceDiv2D</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c15',1)">Detail</a></td>
    </tr>
    
    <tr id='pt15.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt15.1')" >
            pass</a>
    
        <div id='div_pt15.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt15.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt15.1: 
    uniformTensorMesh:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    Yay passed!
    
    
    uniformLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    Yay passed!
    
    
    rotateLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.57e-03   |   3.6062    |  1.8505
    ---------------------------------------------
    And then everyone was happy.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestNodalGrad</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c16',1)">Detail</a></td>
    </tr>
    
    <tr id='pt16.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt16.1')" >
            pass</a>
    
        <div id='div_pt16.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt16.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt16.1: 
    uniformTensorMesh:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    You get a gold star!
    
    
    uniformLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    Testing is important.
    
    
    rotateLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.57e-03   |
       8  |  6.54e-04   |   3.9234    |  1.9721
      16  |  1.80e-04   |   3.6283    |  1.8593
      32  |  4.66e-05   |   3.8703    |  1.9525
    ---------------------------------------------
    You are awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestNodalGrad2D</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c17',1)">Detail</a></td>
    </tr>
    
    <tr id='pt17.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt17.1')" >
            pass</a>
    
        <div id='div_pt17.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt17.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt17.1: 
    uniformTensorMesh:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    You are awesome.
    
    
    uniformLOM:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.56e-03   |
       8  |  6.54e-04   |   3.9232    |  1.9720
      16  |  1.80e-04   |   3.6343    |  1.8617
      32  |  4.64e-05   |   3.8804    |  1.9562
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_optimizers.TestOptimizers</td>
        <td>4</td>
        <td>4</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c18',4)">Detail</a></td>
    </tr>
    
    <tr id='pt18.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_GN_Rosenbrock</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt18.1')" >
            pass</a>
    
        <div id='div_pt18.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt18.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt18.1: =========== Gauss Newton ===========
      #      f      |proj(x-g)-x|  LS  
    -----------------------------------
       0  1.00e+00    2.00e+00      0  
       1  9.53e-01    1.34e+01      2  
       2  4.83e-01    1.19e+00      0  
       3  4.57e-01    1.31e+01      1  
       4  1.89e-01    5.75e-01      0  
       5  1.39e-01    8.15e+00      1  
       6  5.49e-02    5.04e-01      0  
       7  2.91e-02    2.73e+00      1  
       8  9.86e-03    1.37e+00      0  
       9  2.32e-03    1.15e+00      0  
      10  2.38e-04    2.52e-01      0  
      11  4.93e-06    6.73e-02      0  
    ------------------------- STOP! -------------------------
    1 : |fc-fOld| = 2.3305e-04 &lt;= tolF*(1+|f0|) = 2.0000e-01
    1 : |xc-x_last| = 2.8253e-02 &lt;= tolX*(1+|x0|) = 1.0000e-01
    1 : |proj(x-g)-x|    = 6.7282e-02 &lt;= tolG          = 1.0000e-01
    0 : |proj(x-g)-x|    = 6.7282e-02 &lt;= 1e3*eps       = 1.0000e-02
    0 : maxIter   =      20    &lt;= iter          =     11
    ------------------------- DONE! -------------------------
    xopt:  [ 0.99842987  0.99670531]
    x_true:  [ 1.  1.]
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt18.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_GN_quadratic</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt18.2')" >
            pass</a>
    
        <div id='div_pt18.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt18.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt18.2: =========== Gauss Newton ===========
      #      f      |proj(x-g)-x|  LS  
    -----------------------------------
       0  0.00e+00    7.07e+00      0  
       1 -2.50e+01    0.00e+00      0  
    ------------------------- STOP! -------------------------
    0 : |fc-fOld| = 2.5000e+01 &lt;= tolF*(1+|f0|) = 1.0000e-01
    0 : |xc-x_last| = 7.0711e+00 &lt;= tolX*(1+|x0|) = 1.0000e-01
    1 : |proj(x-g)-x|    = 0.0000e+00 &lt;= tolG          = 1.0000e-01
    1 : |proj(x-g)-x|    = 0.0000e+00 &lt;= 1e3*eps       = 1.0000e-02
    0 : maxIter   =      20    &lt;= iter          =      1
    ------------------------- DONE! -------------------------
    xopt:  [ 5.  5.]
    x_true:  [ 5.  5.]
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt18.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ProjGradient_quadratic1Bound</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt18.3')" >
            pass</a>
    
        <div id='div_pt18.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt18.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt18.3: ======================= Projected Gradient =======================
      #      f      |proj(x-g)-x|  LS   itType   aSet    bSet  Comment
    ------------------------------------------------------------------
       0  0.00e+00    2.24e+00      0     SD      0       0           
       1 -8.50e+00    0.00e+00      0     SD      1       1           
    ------------------------- STOP! -------------------------
    0 : |fc-fOld| = 8.5000e+00 &lt;= tolF*(1+|f0|) = 1.0000e-01
    0 : |xc-x_last| = 2.2361e+00 &lt;= tolX*(1+|x0|) = 1.0000e-01
    1 : |proj(x-g)-x|    = 0.0000e+00 &lt;= tolG          = 1.0000e-01
    1 : |proj(x-g)-x|    = 0.0000e+00 &lt;= 1e3*eps       = 1.0000e-02
    0 : maxIter   =      20    &lt;= iter          =      1
    0 : probSize  =      2   &lt;= bindingSet      =      1
    ------------------------- DONE! -------------------------
    xopt:  [ 2. -1.]
    x_true:  [ 2. -1.]
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt18.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ProjGradient_quadraticBounded</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt18.4')" >
            pass</a>
    
        <div id='div_pt18.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt18.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt18.4: ======================= Projected Gradient =======================
      #      f      |proj(x-g)-x|  LS   itType   aSet    bSet  Comment
    ------------------------------------------------------------------
       0  0.00e+00    2.83e+00      0     SD      0       0           
       1 -1.60e+01    0.00e+00      0     SD      2       2           
    ------------------------- STOP! -------------------------
    0 : |fc-fOld| = 1.6000e+01 &lt;= tolF*(1+|f0|) = 1.0000e-01
    0 : |xc-x_last| = 2.8284e+00 &lt;= tolX*(1+|x0|) = 1.0000e-01
    1 : |proj(x-g)-x|    = 0.0000e+00 &lt;= tolG          = 1.0000e-01
    1 : |proj(x-g)-x|    = 0.0000e+00 &lt;= 1e3*eps       = 1.0000e-02
    0 : maxIter   =      20    &lt;= iter          =      1
    1 : probSize  =      2   &lt;= bindingSet      =      2
    ------------------------- DONE! -------------------------
    xopt:  [ 2.  2.]
    x_true:  [ 2.  2.]
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_Solver.TestSolver</td>
        <td>14</td>
        <td>14</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c19',14)">Detail</a></td>
    </tr>
    
    <tr id='pt19.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.9' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.10' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.11' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.12' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.13' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt19.14' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.BasicTensorMeshTests</td>
        <td>7</td>
        <td>7</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c20',7)">Detail</a></td>
    </tr>
    
    <tr id='pt20.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_area_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorCC_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorN_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.TestPoissonEqn</td>
        <td>2</td>
        <td>2</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c21',2)">Detail</a></td>
    </tr>
    
    <tr id='pt21.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderBackward</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt21.1')" >
            pass</a>
    
        <div id='div_pt21.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt21.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt21.1: 
    uniformTensorMesh:  Poisson Equation - Backward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.22e-02   |
      20  |  7.96e-03   |   1.5342    |  1.9182
      24  |  5.59e-03   |   1.4258    |  1.9458
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt21.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderForward</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt21.2')" >
            pass</a>
    
        <div id='div_pt21.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt21.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt21.2: 
    uniformTensorMesh:  Poisson Equation - Forward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.43e+00   |
      20  |  9.35e-01   |   1.5271    |  1.8974
      24  |  6.58e-01   |   1.4223    |  1.9320
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_utils.TestCheckDerivative</td>
        <td>3</td>
        <td>3</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c22',3)">Detail</a></td>
    </tr>
    
    <tr id='pt22.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFail</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt22.1')" >
            pass</a>
    
        <div id='div_pt22.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt22.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt22.1: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.967e-01		3.904e-01		nan
    1	1.00e-02	1.941e-02		3.880e-02		1.003
    2	1.00e-03	1.939e-03		3.877e-03		1.000
    3	1.00e-04	1.938e-04		3.876e-04		1.000
    4	1.00e-05	1.938e-05		3.876e-05		1.000
    5	1.00e-06	1.938e-06		3.876e-06		1.000
    6	1.00e-07	1.938e-07		3.876e-07		1.000
    *********************************************************
    &lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt; FAIL! &gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;
    *********************************************************
    Coffee break?
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt22.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFunction</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt22.2')" >
            pass</a>
    
        <div id='div_pt22.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt22.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt22.2: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.210e-01		1.399e-03		nan
    1	1.00e-02	1.215e-02		1.279e-05		2.039
    2	1.00e-03	1.216e-03		1.269e-07		2.003
    3	1.00e-04	1.216e-04		1.268e-09		2.000
    4	1.00e-05	1.216e-05		1.268e-11		2.000
    5	1.00e-06	1.216e-06		1.267e-13		2.000
    6	1.00e-07	1.216e-07		1.249e-15		2.006
    ========================= PASS! =========================
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt22.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simplePass</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt22.3')" >
            pass</a>
    
        <div id='div_pt22.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt22.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt22.3: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.414e-01		4.527e-03		nan
    1	1.00e-02	2.393e-02		5.089e-05		1.949
    2	1.00e-03	2.390e-03		5.150e-07		1.995
    3	1.00e-04	2.390e-04		5.156e-09		1.999
    4	1.00e-05	2.390e-05		5.157e-11		2.000
    5	1.00e-06	2.390e-06		5.157e-13		2.000
    6	1.00e-07	2.390e-07		5.174e-15		1.999
    ========================= PASS! =========================
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_utils.TestSequenceFunctions</td>
        <td>8</td>
        <td>8</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c23',8)">Detail</a></td>
    </tr>
    
    <tr id='pt23.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_invXXXBlockDiagonal</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc2</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc3</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt23.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='total_row'>
        <td>Total</td>
        <td>117</td>
        <td>117</td>
        <td>0</td>
        <td>0</td>
        <td>&nbsp;</td>
    </tr>
    </table>
    
