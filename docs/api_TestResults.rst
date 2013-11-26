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
                var divTid = document.getElementById('div_'+tid);
                if(divTid !== null){divTid.style.display = 'none';}
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
    <p class='attribute'><strong>Start Time:</strong> 2013-11-21 11:59:36</p>
    <p class='attribute'><strong>Duration:</strong> 0:00:32.074343</p>
    <p class='attribute'><strong>Status:</strong> Pass 124</p>
    
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
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt3.1')" >
            pass</a>
    
        <div id='div_pt3.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt3.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt3.1: Setting bfgsH0 to the inverse of the modelObj2Deriv.
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
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
            
    pt3.2: Setting bfgsH0 to the inverse of the modelObj2Deriv.
    ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	6.680e+01		6.164e+01		nan
    1	1.00e-02	1.154e+00		6.381e-01		1.985
    2	1.00e-03	5.798e-02		6.401e-03		1.999
    3	1.00e-04	5.222e-03		6.403e-05		2.000
    4	1.00e-05	5.164e-04		6.403e-07		2.000
    5	1.00e-06	5.159e-05		6.403e-09		2.000
    6	1.00e-07	5.158e-06		6.377e-11		2.002
    ========================= PASS! =========================
    You get a gold star!
    
    
    
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
            
    pt3.3: Setting bfgsH0 to the inverse of the modelObj2Deriv.
    ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.203e-01		1.785e-02		nan
    1	1.00e-02	1.175e-02		1.805e-04		1.995
    2	1.00e-03	1.173e-03		1.805e-06		2.000
    3	1.00e-04	1.173e-04		1.805e-08		2.000
    4	1.00e-05	1.173e-05		1.805e-10		2.000
    5	1.00e-06	1.173e-06		1.805e-12		2.000
    6	1.00e-07	1.173e-07		1.915e-14		1.975
    ========================= PASS! =========================
    The test be workin!
    
    
    
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
            
    pt3.4: Setting bfgsH0 to the inverse of the modelObj2Deriv.
    ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	6.895e+00		6.894e+00		nan
    1	1.00e-02	6.903e-02		6.894e-02		2.000
    2	1.00e-03	6.982e-04		6.894e-04		2.000
    3	1.00e-04	7.769e-06		6.894e-06		2.000
    4	1.00e-05	1.564e-07		6.894e-08		2.000
    5	1.00e-06	9.431e-09		6.894e-10		2.000
    6	1.00e-07	8.810e-10		6.895e-12		2.000
    ========================= PASS! =========================
    Go Test Go!
    
    
    
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
    0	1.00e-01	1.936e-01		7.633e-17		nan
    1	1.00e-02	1.936e-02		7.268e-17		0.021
    2	1.00e-03	1.936e-03		6.545e-17		0.046
    3	1.00e-04	1.936e-04		8.778e-17		-0.127
    4	1.00e-05	1.936e-05		6.714e-17		0.116
    5	1.00e-06	1.936e-06		5.514e-17		0.086
    6	1.00e-07	1.936e-07		8.359e-17		-0.181
    ========================= PASS! =========================
    The test be workin!
    
    
    
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
    0	1.00e-01	3.925e-01		6.586e-02		nan
    1	1.00e-02	4.517e-02		6.586e-04		2.000
    2	1.00e-03	4.577e-03		6.586e-06		2.000
    3	1.00e-04	4.582e-04		6.586e-08		2.000
    4	1.00e-05	4.583e-05		6.586e-10		2.000
    5	1.00e-06	4.583e-06		6.587e-12		2.000
    6	1.00e-07	4.583e-07		6.827e-14		1.984
    ========================= PASS! =========================
    Not just a pretty face Rowan
    
    
    
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
       8  |  2.91e-01   |
      16  |  6.70e-02   |   4.3512    |  2.1214
      32  |  1.48e-02   |   4.5346    |  2.1810
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    randomTensorMesh:  Interpolation 1D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.81e-01   |
      16  |  9.23e-02   |   5.2190    |  1.9811
      32  |  1.61e-02   |   5.7473    |  3.1176
    ---------------------------------------------
    You get a gold star!
    
    
    
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
       8  |  2.46e-01   |
      16  |  6.96e-02   |   3.5355    |  1.8219
      32  |  1.88e-02   |   3.6930    |  1.8848
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    randomTensorMesh:  Interpolation 1D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  6.95e-01   |
      16  |  1.12e-01   |   6.2065    |  2.7830
      32  |  4.60e-02   |   2.4348    |  2.0359
    ---------------------------------------------
    The test be workin!
    
    
    
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
       8  |  7.59e-02   |
      16  |  1.90e-02   |   3.9894    |  1.9962
      32  |  4.72e-03   |   4.0321    |  2.0115
      64  |  1.20e-03   |   3.9485    |  1.9813
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation 2D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.52e-01   |
      16  |  2.90e-02   |   5.2472    |  1.9020
      32  |  9.09e-03   |   3.1868    |  1.4098
      64  |  2.68e-03   |   3.3935    |  2.1529
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
       8  |  6.57e-02   |
      16  |  1.88e-02   |   3.4893    |  1.8030
      32  |  4.79e-03   |   3.9343    |  1.9761
      64  |  1.17e-03   |   4.1029    |  2.0366
    ---------------------------------------------
    Go Test Go!
    
    
    randomTensorMesh:  Interpolation 2D: Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.16e-01   |
      16  |  3.68e-02   |   3.1394    |  1.5218
      32  |  8.77e-03   |   4.1974    |  3.1728
      64  |  3.14e-03   |   2.7963    |  1.7619
    ---------------------------------------------
    Testing is important.
    
    
    
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
       8  |  7.04e-02   |
      16  |  1.88e-02   |   3.7494    |  1.9067
      32  |  4.55e-03   |   4.1251    |  2.0444
      64  |  1.16e-03   |   3.9315    |  1.9751
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation 2D: Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.99e-01   |
      16  |  8.02e-02   |   2.4773    |  4.5721
      32  |  1.30e-02   |   6.1949    |  1.7850
      64  |  3.95e-03   |   3.2812    |  1.9890
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
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
       8  |  7.59e-02   |
      16  |  1.90e-02   |   3.9894    |  1.9962
      32  |  4.72e-03   |   4.0321    |  2.0115
      64  |  1.20e-03   |   3.9485    |  1.9813
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation 2D: Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  5.31e-02   |
      16  |  4.42e-02   |   1.2007    |  0.2070
      32  |  1.05e-02   |   4.2086    |  2.5806
      64  |  1.80e-03   |   5.8391    |  2.4462
    ---------------------------------------------
    The test be workin!
    
    
    
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
       8  |  7.61e-02   |
      16  |  1.92e-02   |   3.9697    |  1.9890
      32  |  4.76e-03   |   4.0235    |  2.0084
      64  |  1.15e-03   |   4.1348    |  2.0478
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    randomTensorMesh:  Interpolation 2D: Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.37e-01   |
      16  |  3.12e-02   |   4.3873    |  2.7317
      32  |  7.83e-03   |   3.9895    |  2.3562
      64  |  2.26e-03   |   3.4691    |  1.3948
    ---------------------------------------------
    Happy little convergence test!
    
    
    
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
       8  |  6.57e-02   |
      16  |  1.88e-02   |   3.4893    |  1.8030
      32  |  4.79e-03   |   3.9343    |  1.9761
      64  |  1.17e-03   |   4.1029    |  2.0366
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation 2D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.16e-01   |
      16  |  3.36e-02   |   6.4220    |  1.9825
      32  |  2.07e-02   |   1.6177    |  1.1560
      64  |  3.05e-03   |   6.8009    |  2.5482
    ---------------------------------------------
    The test be workin!
    
    
    
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
       8  |  7.38e-02   |
      16  |  1.70e-02   |   4.3363    |  2.1165
      32  |  4.67e-03   |   3.6472    |  1.8668
      64  |  1.12e-03   |   4.1563    |  2.0553
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.39e-01   |
      16  |  2.83e-02   |   8.4410    |  3.3650
      32  |  8.35e-03   |   3.3902    |  1.4106
      64  |  2.59e-03   |   3.2260    |  1.7120
    ---------------------------------------------
    You are awesome.
    
    
    
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
       8  |  6.97e-02   |
      16  |  1.88e-02   |   3.7094    |  1.8912
      32  |  4.68e-03   |   4.0208    |  2.0075
      64  |  1.17e-03   |   3.9957    |  1.9985
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    randomTensorMesh:  Interpolation Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.13e-01   |
      16  |  5.12e-02   |   4.1677    |  1.3954
      32  |  1.99e-02   |   2.5661    |  1.5524
      64  |  3.48e-03   |   5.7313    |  2.1382
    ---------------------------------------------
    You get a gold star!
    
    
    
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
       8  |  7.04e-02   |
      16  |  1.86e-02   |   3.7826    |  1.9194
      32  |  4.79e-03   |   3.8833    |  1.9573
      64  |  1.17e-03   |   4.1003    |  2.0357
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.43e-01   |
      16  |  5.66e-02   |   4.2904    |  1.4531
      32  |  1.55e-02   |   3.6614    |  2.1536
      64  |  3.27e-03   |   4.7300    |  2.2808
    ---------------------------------------------
    That was easy!
    
    
    
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
       8  |  6.98e-02   |
      16  |  1.81e-02   |   3.8612    |  1.9491
      32  |  4.70e-03   |   3.8479    |  1.9441
      64  |  1.20e-03   |   3.9268    |  1.9733
    ---------------------------------------------
    And then everyone was happy.
    
    
    randomTensorMesh:  Interpolation Ez
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.77e-02   |
      16  |  4.93e-02   |   1.9820    |  2.0322
      32  |  1.34e-02   |   3.6656    |  1.4108
      64  |  4.90e-03   |   2.7436    |  1.4520
    ---------------------------------------------
    Yay passed!
    
    
    
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
       8  |  7.38e-02   |
      16  |  1.70e-02   |   4.3363    |  2.1165
      32  |  4.67e-03   |   3.6472    |  1.8668
      64  |  1.12e-03   |   4.1563    |  2.0553
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.51e-01   |
      16  |  2.31e-02   |   6.5343    |  3.0196
      32  |  1.32e-02   |   1.7456    |  0.8096
      64  |  2.24e-03   |   5.8991    |  2.1039
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
       8  |  7.61e-02   |
      16  |  1.92e-02   |   3.9630    |  1.9866
      32  |  4.81e-03   |   3.9962    |  1.9986
      64  |  1.20e-03   |   4.0135    |  2.0049
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.80e-01   |
      16  |  4.05e-02   |   4.4477    |  1.4993
      32  |  1.59e-02   |   2.5460    |  1.1426
      64  |  2.54e-03   |   6.2522    |  2.3141
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
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
       8  |  7.51e-02   |
      16  |  1.82e-02   |   4.1301    |  2.0462
      32  |  4.16e-03   |   4.3667    |  2.1266
      64  |  1.13e-03   |   3.6758    |  1.8781
    ---------------------------------------------
    And then everyone was happy.
    
    
    randomTensorMesh:  Interpolation Fz
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.32e-01   |
      16  |  5.78e-02   |   2.2902    |  1.6547
      32  |  9.69e-03   |   5.9708    |  2.9133
      64  |  4.41e-03   |   2.1983    |  1.1867
    ---------------------------------------------
    Well done Rowan!
    
    
    
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
       8  |  6.97e-02   |
      16  |  1.88e-02   |   3.7094    |  1.8912
      32  |  4.68e-03   |   4.0208    |  2.0075
      64  |  1.17e-03   |   3.9957    |  1.9985
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.50e-02   |
      16  |  7.65e-02   |   0.9805    |  -0.0299
      32  |  1.75e-02   |   4.3794    |  2.0088
      64  |  4.46e-03   |   3.9163    |  1.9951
    ---------------------------------------------
    And then everyone was happy.
    
    
    
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
    Testing is important.
    
    
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
    Not just a pretty face Rowan
    
    
    
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
    And then everyone was happy.
    
    
    uniformLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.29e-04   |
      32  |  1.57e-04   |   3.9978    |  1.9992
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    rotateLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.08e-04   |
      32  |  7.07e-05   |   4.3564    |  2.1231
    ---------------------------------------------
    Well done Rowan!
    
    
    
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
    And then everyone was happy.
    
    
    uniformLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.99e-03   |
      32  |  1.75e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  7.70e-03   |
      32  |  1.94e-03   |   3.9622    |  1.9863
    ---------------------------------------------
    Go Test Go!
    
    
    
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
    You deserve a pat on the back!
    
    
    uniformLOM:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.68e-03   |
      32  |  6.69e-04   |   3.9982    |  1.9993
    ---------------------------------------------
    You are awesome.
    
    
    rotateLOM:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.15e-03   |
      32  |  5.25e-04   |   4.0845    |  2.0302
    ---------------------------------------------
    The test be workin!
    
    
    
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
    Not just a pretty face Rowan
    
    
    uniformLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.79e-03   |
      32  |  1.70e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  7.49e-03   |
      32  |  1.89e-03   |   3.9617    |  1.9861
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
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
    You are awesome.
    
    
    uniformLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.10e-03   |
      32  |  7.74e-04   |   3.9981    |  1.9993
    ---------------------------------------------
    You are awesome.
    
    
    rotateLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.54e-03   |
      32  |  6.23e-04   |   4.0741    |  2.0265
    ---------------------------------------------
    Testing is important.
    
    
    
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
    Happy little convergence test!
    
    
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
    You deserve a pat on the back!
    
    
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
    Go Test Go!
    
    
    
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
    You get a gold star!
    
    
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
    Yay passed!
    
    
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
    Well done Rowan!
    
    
    
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
    And then everyone was happy.
    
    
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
    Awesome, Rowan, just awesome.
    
    
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
    You get a gold star!
    
    
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
    Awesome, Rowan, just awesome.
    
    
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
    Well done Rowan!
    
    
    
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
    And then everyone was happy.
    
    
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
    Go Test Go!
    
    
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
    Once upon a time, a happy little test passed.
    
    
    
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
    Go Test Go!
    
    
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
    You get a gold star!
    
    
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
    Testing is important.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestAveraging2D</td>
        <td>6</td>
        <td>6</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c11',6)">Detail</a></td>
    </tr>
    
    <tr id='pt11.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderCC2F</div></td>
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
    uniformTensorMesh:  Averaging 2D: CC2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.25e-01   |
       8  |  6.25e-02   |   1.9961    |  0.9972
      16  |  3.12e-02   |   1.9990    |  0.9993
      32  |  1.56e-02   |   1.9998    |  0.9998
    ---------------------------------------------
    You are awesome.
    
    
    uniformLOM:  Averaging 2D: CC2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.25e-01   |
       8  |  6.25e-02   |   1.9961    |  0.9972
      16  |  3.12e-02   |   1.9990    |  0.9993
      32  |  1.56e-02   |   1.9998    |  0.9998
    ---------------------------------------------
    And then everyone was happy.
    
    
    rotateLOM:  Averaging 2D: CC2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.28e-01   |
       8  |  7.22e-02   |   1.7667    |  0.8210
      16  |  3.81e-02   |   1.8958    |  0.9228
      32  |  1.95e-02   |   1.9507    |  0.9640
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderE2CC</div></td>
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
    uniformTensorMesh:  Averaging 2D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    That was easy!
    
    
    uniformLOM:  Averaging 2D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    And then everyone was happy.
    
    
    rotateLOM:  Averaging 2D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.88e-03   |
       8  |  1.82e-03   |   3.7846    |  1.9202
      16  |  4.66e-04   |   3.8970    |  1.9624
      32  |  1.18e-04   |   3.9552    |  1.9837
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderF2CC</div></td>
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
    uniformTensorMesh:  Averaging 2D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  6.87e-03   |
       8  |  1.76e-03   |   3.8978    |  1.9627
      16  |  4.45e-04   |   3.9561    |  1.9841
      32  |  1.12e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    You are awesome.
    
    
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
    Happy little convergence test!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2CC</div></td>
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
    uniformTensorMesh:  Averaging 2D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.37e-02   |
       8  |  3.52e-03   |   3.8978    |  1.9627
      16  |  8.90e-04   |   3.9561    |  1.9841
      32  |  2.24e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    That was easy!
    
    
    uniformLOM:  Averaging 2D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.37e-02   |
       8  |  3.52e-03   |   3.8978    |  1.9627
      16  |  8.90e-04   |   3.9561    |  1.9841
      32  |  2.24e-04   |   3.9799    |  1.9927
    ---------------------------------------------
    Happy little convergence test!
    
    
    rotateLOM:  Averaging 2D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.38e-02   |
       8  |  3.64e-03   |   3.7900    |  1.9222
      16  |  9.33e-04   |   3.8980    |  1.9627
      32  |  2.36e-04   |   3.9577    |  1.9847
    ---------------------------------------------
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2E</div></td>
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
    uniformTensorMesh:  Averaging 2D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Averaging 2D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    Happy little convergence test!
    
    
    rotateLOM:  Averaging 2D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.02e-02   |
       8  |  2.73e-03   |   3.7288    |  1.8987
      16  |  7.34e-04   |   3.7229    |  1.8964
      32  |  1.86e-04   |   3.9505    |  1.9820
    ---------------------------------------------
    And then everyone was happy.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt11.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2F</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt11.6')" >
            pass</a>
    
        <div id='div_pt11.6' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt11.6').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt11.6: 
    uniformTensorMesh:  Averaging 2D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    Well done Rowan!
    
    
    uniformLOM:  Averaging 2D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  7.74e-03   |
       8  |  1.95e-03   |   3.9727    |  1.9901
      16  |  4.88e-04   |   3.9932    |  1.9975
      32  |  1.22e-04   |   3.9983    |  1.9994
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    rotateLOM:  Averaging 2D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.02e-02   |
       8  |  2.73e-03   |   3.7288    |  1.8987
      16  |  7.34e-04   |   3.7229    |  1.8964
      32  |  1.86e-04   |   3.9505    |  1.9820
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestAveraging3D</td>
        <td>6</td>
        <td>6</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c12',6)">Detail</a></td>
    </tr>
    
    <tr id='pt12.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderCC2F</div></td>
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
    uniformTensorMesh:  Averaging 3D: CC2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.19e-01   |
       8  |  1.65e-01   |   1.9394    |  0.9556
      16  |  8.36e-02   |   1.9692    |  0.9776
      32  |  4.21e-02   |   1.9845    |  0.9888
    ---------------------------------------------
    Yay passed!
    
    
    uniformLOM:  Averaging 3D: CC2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.19e-01   |
       8  |  1.65e-01   |   1.9394    |  0.9556
      16  |  8.36e-02   |   1.9692    |  0.9776
      32  |  4.21e-02   |   1.9845    |  0.9888
    ---------------------------------------------
    Well done Rowan!
    
    
    rotateLOM:  Averaging 3D: CC2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  3.57e-01   |
       8  |  1.92e-01   |   1.8621    |  0.8969
      16  |  9.95e-02   |   1.9268    |  0.9462
      32  |  4.98e-02   |   1.9972    |  0.9980
    ---------------------------------------------
    You get a gold star!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderE2CC</div></td>
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
    uniformTensorMesh:  Averaging 3D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  8.53e-03   |
       8  |  2.47e-03   |   3.4463    |  1.7851
      16  |  6.63e-04   |   3.7311    |  1.8996
      32  |  1.71e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Averaging 3D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  8.53e-03   |
       8  |  2.47e-03   |   3.4463    |  1.7851
      16  |  6.63e-04   |   3.7311    |  1.8996
      32  |  1.71e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Testing is important.
    
    
    rotateLOM:  Averaging 3D: E2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  9.08e-03   |
       8  |  2.55e-03   |   3.5588    |  1.8314
      16  |  7.77e-04   |   3.2852    |  1.7160
      32  |  2.08e-04   |   3.7415    |  1.9036
    ---------------------------------------------
    You are awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderF2CC</div></td>
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
    uniformTensorMesh:  Averaging 3D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.26e-03   |
       8  |  1.24e-03   |   3.4463    |  1.7851
      16  |  3.32e-04   |   3.7311    |  1.8996
      32  |  8.57e-05   |   3.8674    |  1.9514
    ---------------------------------------------
    Yay passed!
    
    
    uniformLOM:  Averaging 3D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.26e-03   |
       8  |  1.24e-03   |   3.4463    |  1.7851
      16  |  3.32e-04   |   3.7311    |  1.8996
      32  |  8.57e-05   |   3.8674    |  1.9514
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Averaging 3D: F2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.54e-03   |
       8  |  1.28e-03   |   3.5589    |  1.8314
      16  |  3.88e-04   |   3.2842    |  1.7155
      32  |  1.04e-04   |   3.7417    |  1.9037
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2CC</div></td>
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
    uniformTensorMesh:  Averaging 3D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.28e-02   |
       8  |  3.71e-03   |   3.4463    |  1.7851
      16  |  9.95e-04   |   3.7311    |  1.8996
      32  |  2.57e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    Well done Rowan!
    
    
    uniformLOM:  Averaging 3D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.28e-02   |
       8  |  3.71e-03   |   3.4463    |  1.7851
      16  |  9.95e-04   |   3.7311    |  1.8996
      32  |  2.57e-04   |   3.8674    |  1.9514
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Averaging 3D: N2CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.36e-02   |
       8  |  3.83e-03   |   3.5589    |  1.8314
      16  |  1.17e-03   |   3.2862    |  1.7164
      32  |  3.11e-04   |   3.7413    |  1.9036
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2E</div></td>
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
    uniformTensorMesh:  Averaging 3D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.88e-02   |
       8  |  4.99e-03   |   3.7613    |  1.9112
      16  |  1.29e-03   |   3.8779    |  1.9553
      32  |  3.27e-04   |   3.9382    |  1.9775
    ---------------------------------------------
    Yay passed!
    
    
    uniformLOM:  Averaging 3D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.88e-02   |
       8  |  4.99e-03   |   3.7613    |  1.9112
      16  |  1.29e-03   |   3.8779    |  1.9553
      32  |  3.27e-04   |   3.9382    |  1.9775
    ---------------------------------------------
    Yay passed!
    
    
    rotateLOM:  Averaging 3D: N2E
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.41e-02   |
       8  |  6.11e-03   |   3.9523    |  1.9827
      16  |  1.70e-03   |   3.5908    |  1.8443
      32  |  4.41e-04   |   3.8566    |  1.9473
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt12.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderN2F</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt12.6')" >
            pass</a>
    
        <div id='div_pt12.6' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt12.6').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt12.6: 
    uniformTensorMesh:  Averaging 3D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.78e-02   |
       8  |  4.87e-03   |   3.6557    |  1.8701
      16  |  1.27e-03   |   3.8285    |  1.9368
      32  |  3.25e-04   |   3.9144    |  1.9688
    ---------------------------------------------
    Go Test Go!
    
    
    uniformLOM:  Averaging 3D: N2F
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  1.78e-02   |
       8  |  4.87e-03   |   3.6557    |  1.8701
      16  |  1.27e-03   |   3.8285    |  1.9368
      32  |  3.25e-04   |   3.9144    |  1.9688
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
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
        <td>test_operators.TestCellGrad1D_InhomogeneousDirichlet</td>
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
    uniformTensorMesh:  Cell Grad 1D - Dirichlet
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.22e+00   |
      16  |  6.15e-01   |   1.9808    |  0.9861
      32  |  3.08e-01   |   1.9952    |  0.9965
      64  |  1.54e-01   |   1.9988    |  0.9991
    ---------------------------------------------
    The test be workin!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestCellGrad2D_Dirichlet</td>
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
    uniformTensorMesh:  Cell Grad 2D - Dirichlet
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.48e-01   |
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
      64  |  2.52e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    And then everyone was happy.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestCellGrad2D_Neumann</td>
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
    uniformTensorMesh:  Cell Grad 2D - Neumann
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.48e-01   |
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
      64  |  2.52e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    You get a gold star!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestCellGrad3D_Dirichlet</td>
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
    uniformTensorMesh:  Cell Grad 3D - Dirichlet
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.37e-01   |
      16  |  3.88e-02   |   3.5288    |  1.8192
      32  |  9.99e-03   |   3.8795    |  1.9559
    ---------------------------------------------
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestCellGrad3D_Neumann</td>
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
    uniformTensorMesh:  Cell Grad 3D - Neumann
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.37e-01   |
      16  |  3.88e-02   |   3.5288    |  1.8192
      32  |  9.99e-03   |   3.8795    |  1.9559
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    
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
        <td><a href="javascript:showClassDetail('c18',1)">Detail</a></td>
    </tr>
    
    <tr id='pt18.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
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
            
    pt18.1: 
    uniformTensorMesh:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.71e-01   |
       8  |  8.86e-02   |   5.3170    |  2.4106
      16  |  1.70e-02   |   5.2040    |  2.3796
      32  |  3.77e-03   |   4.5126    |  2.1740
    ---------------------------------------------
    Awesome, Rowan, just awesome.
    
    
    
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
        <td><a href="javascript:showClassDetail('c19',1)">Detail</a></td>
    </tr>
    
    <tr id='pt19.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt19.1')" >
            pass</a>
    
        <div id='div_pt19.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt19.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt19.1: 
    uniformTensorMesh:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    uniformLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    Well done Rowan!
    
    
    rotateLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.57e-03   |   3.6062    |  1.8505
    ---------------------------------------------
    Go Test Go!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_operators.TestFaceDiv3D</td>
        <td>1</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c20',1)">Detail</a></td>
    </tr>
    
    <tr id='pt20.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt20.1')" >
            pass</a>
    
        <div id='div_pt20.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt20.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt20.1: 
    uniformTensorMesh:  Face Divergence 3D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Yay passed!
    
    
    uniformLOM:  Face Divergence 3D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    You are awesome.
    
    
    rotateLOM:  Face Divergence 3D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.08e-03   |
      16  |  9.53e-04   |   9.5374    |  3.2536
      32  |  2.75e-04   |   3.4594    |  1.7905
    ---------------------------------------------
    You are awesome.
    
    
    
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
        <td><a href="javascript:showClassDetail('c21',1)">Detail</a></td>
    </tr>
    
    <tr id='pt21.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
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
    uniformTensorMesh:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    That was easy!
    
    
    uniformLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
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
        <td><a href="javascript:showClassDetail('c22',1)">Detail</a></td>
    </tr>
    
    <tr id='pt22.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
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
            
    pt22.1: 
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
    You are awesome.
    
    
    rotateLOM:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.56e-03   |
       8  |  6.54e-04   |   3.9232    |  1.9720
      16  |  1.80e-04   |   3.6343    |  1.8617
      32  |  4.64e-05   |   3.8804    |  1.9562
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
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
        <td><a href="javascript:showClassDetail('c23',4)">Detail</a></td>
    </tr>
    
    <tr id='pt23.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_GN_Rosenbrock</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt23.1')" >
            pass</a>
    
        <div id='div_pt23.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt23.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt23.1: =========== Gauss Newton ===========
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
    
    <tr id='pt23.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_GN_quadratic</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt23.2')" >
            pass</a>
    
        <div id='div_pt23.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt23.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt23.2: =========== Gauss Newton ===========
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
    
    <tr id='pt23.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ProjGradient_quadratic1Bound</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt23.3')" >
            pass</a>
    
        <div id='div_pt23.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt23.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt23.3: ========================== Projected Gradient ==========================
      #      f      |proj(x-g)-x|  LS   itType   aSet    bSet    Comment   
    -----------------------------------------------------------------------
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
    
    <tr id='pt23.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ProjGradient_quadraticBounded</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt23.4')" >
            pass</a>
    
        <div id='div_pt23.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt23.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt23.4: ========================== Projected Gradient ==========================
      #      f      |proj(x-g)-x|  LS   itType   aSet    bSet    Comment   
    -----------------------------------------------------------------------
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
        <td><a href="javascript:showClassDetail('c24',14)">Detail</a></td>
    </tr>
    
    <tr id='pt24.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.9' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.10' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.11' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.12' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.13' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt24.14' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.BasicTensorMeshTests</td>
        <td>7</td>
        <td>7</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c25',7)">Detail</a></td>
    </tr>
    
    <tr id='pt25.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_area_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt25.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt25.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt25.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorCC_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt25.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorN_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt25.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt25.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.TestPoissonEqn</td>
        <td>2</td>
        <td>2</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c26',2)">Detail</a></td>
    </tr>
    
    <tr id='pt26.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderBackward</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt26.1')" >
            pass</a>
    
        <div id='div_pt26.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt26.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt26.1: 
    uniformTensorMesh:  Poisson Equation - Backward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.22e-02   |
      20  |  7.96e-03   |   1.5342    |  1.9182
      24  |  5.59e-03   |   1.4258    |  1.9458
    ---------------------------------------------
    That was easy!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt26.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderForward</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt26.2')" >
            pass</a>
    
        <div id='div_pt26.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt26.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt26.2: 
    uniformTensorMesh:  Poisson Equation - Forward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.43e+00   |
      20  |  9.35e-01   |   1.5271    |  1.8974
      24  |  6.58e-01   |   1.4223    |  1.9320
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
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
        <td><a href="javascript:showClassDetail('c27',3)">Detail</a></td>
    </tr>
    
    <tr id='pt27.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFail</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt27.1')" >
            pass</a>
    
        <div id='div_pt27.1' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt27.1').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt27.1: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.813e-01		3.602e-01		nan
    1	1.00e-02	1.792e-02		3.581e-02		1.003
    2	1.00e-03	1.789e-03		3.578e-03		1.000
    3	1.00e-04	1.789e-04		3.578e-04		1.000
    4	1.00e-05	1.789e-05		3.578e-05		1.000
    5	1.00e-06	1.789e-06		3.578e-06		1.000
    6	1.00e-07	1.789e-07		3.578e-07		1.000
    *********************************************************
    &lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt; FAIL! &gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;
    *********************************************************
    Thankfully, persistence is a great substitute for talent.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt27.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFunction</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt27.2')" >
            pass</a>
    
        <div id='div_pt27.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt27.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt27.2: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.507e-01		1.960e-02		nan
    1	1.00e-02	2.577e-02		1.872e-04		2.020
    2	1.00e-03	2.584e-03		1.863e-06		2.002
    3	1.00e-04	2.584e-04		1.862e-08		2.000
    4	1.00e-05	2.584e-05		1.862e-10		2.000
    5	1.00e-06	2.584e-06		1.862e-12		2.000
    6	1.00e-07	2.584e-07		1.859e-14		2.001
    ========================= PASS! =========================
    The test be workin!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt27.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simplePass</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt27.3')" >
            pass</a>
    
        <div id='div_pt27.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt27.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt27.3: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	6.038e-02		4.490e-04		nan
    1	1.00e-02	6.069e-03		4.327e-06		2.016
    2	1.00e-03	6.072e-04		4.311e-08		2.002
    3	1.00e-04	6.072e-05		4.310e-10		2.000
    4	1.00e-05	6.072e-06		4.309e-12		2.000
    5	1.00e-06	6.072e-07		4.307e-14		2.000
    6	1.00e-07	6.072e-08		4.903e-16		1.944
    ========================= PASS! =========================
    Not just a pretty face Rowan
    
    
    
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
        <td><a href="javascript:showClassDetail('c28',8)">Detail</a></td>
    </tr>
    
    <tr id='pt28.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_invXXXBlockDiagonal</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc2</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc3</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt28.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='total_row'>
        <td>Total</td>
        <td>124</td>
        <td>124</td>
        <td>0</td>
        <td>0</td>
        <td>&nbsp;</td>
    </tr>
    </table>
    
