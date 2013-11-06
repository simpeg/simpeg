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
    <p class='attribute'><strong>Start Time:</strong> 2013-11-06 11:11:28</p>
    <p class='attribute'><strong>Duration:</strong> 0:00:31.165229</p>
    <p class='attribute'><strong>Status:</strong> Pass 99</p>
    
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
    0	1.00e-01	3.539e+01		3.808e+01		nan
    1	1.00e-02	1.088e-01		3.784e-01		2.003
    2	1.00e-03	2.318e-02		3.783e-03		2.000
    3	1.00e-04	2.659e-03		3.782e-05		2.000
    4	1.00e-05	2.693e-04		3.782e-07		2.000
    5	1.00e-06	2.696e-05		3.787e-09		2.000
    6	1.00e-07	2.696e-06		4.574e-11		1.918
    ========================= PASS! =========================
    You are awesome.
    
    
    
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
    0	1.00e-01	5.058e-02		4.612e-03		nan
    1	1.00e-02	5.059e-03		4.602e-05		2.001
    2	1.00e-03	5.060e-04		4.600e-07		2.000
    3	1.00e-04	5.060e-05		4.600e-09		2.000
    4	1.00e-05	5.060e-06		4.600e-11		2.000
    5	1.00e-06	5.060e-07		4.619e-13		1.998
    6	1.00e-07	5.060e-08		4.783e-15		1.985
    ========================= PASS! =========================
    You are awesome.
    
    
    
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
    0	1.00e-01	7.509e+00		7.507e+00		nan
    1	1.00e-02	7.529e-02		7.507e-02		2.000
    2	1.00e-03	7.725e-04		7.507e-04		2.000
    3	1.00e-04	9.680e-06		7.507e-06		2.000
    4	1.00e-05	2.924e-07		7.507e-08		2.000
    5	1.00e-06	2.248e-08		7.507e-10		2.000
    6	1.00e-07	2.181e-09		7.507e-12		2.000
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
    0	1.00e-01	2.750e-01		6.208e-17		nan
    1	1.00e-02	2.750e-02		6.731e-17		-0.035
    2	1.00e-03	2.750e-03		6.483e-17		0.016
    3	1.00e-04	2.750e-04		5.624e-17		0.062
    4	1.00e-05	2.750e-05		6.905e-17		-0.089
    5	1.00e-06	2.750e-06		3.779e-17		0.262
    6	1.00e-07	2.750e-07		5.028e-17		-0.124
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
    0	1.00e-01	5.097e-01		3.406e-02		nan
    1	1.00e-02	4.790e-02		3.406e-04		2.000
    2	1.00e-03	4.760e-03		3.406e-06		2.000
    3	1.00e-04	4.757e-04		3.406e-08		2.000
    4	1.00e-05	4.756e-05		3.406e-10		2.000
    5	1.00e-06	4.756e-06		3.409e-12		2.000
    6	1.00e-07	4.756e-07		3.712e-14		1.963
    ========================= PASS! =========================
    Once upon a time, a happy little test passed.
    
    
    
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
       8  |  2.73e-01   |
      16  |  7.43e-02   |   3.6706    |  1.8760
      32  |  1.68e-02   |   4.4243    |  2.1454
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 1D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.10e-01   |
      16  |  1.03e-01   |   3.9943    |  2.3138
      32  |  2.38e-02   |   4.3129    |  2.1747
    ---------------------------------------------
    Go Test Go!
    
    
    
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
       8  |  2.67e-01   |
      16  |  6.20e-02   |   4.3031    |  2.1054
      32  |  1.74e-02   |   3.5710    |  1.8363
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 1D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  5.93e-01   |
      16  |  6.78e-02   |   8.7462    |  3.4908
      32  |  3.76e-02   |   1.8044    |  0.9007
    ---------------------------------------------
    You get a gold star!
    
    
    
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
       8  |  7.05e-02   |
      16  |  1.77e-02   |   3.9862    |  1.9950
      32  |  4.41e-03   |   4.0153    |  2.0055
      64  |  1.17e-03   |   3.7677    |  1.9137
    ---------------------------------------------
    The test be workin!
    
    
    randomTensorMesh:  Interpolation 2D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.89e-01   |
      16  |  4.17e-02   |   4.5168    |  1.7836
      32  |  9.85e-03   |   4.2370    |  3.1893
      64  |  2.20e-03   |   4.4807    |  2.0262
    ---------------------------------------------
    And then everyone was happy.
    
    
    
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
       8  |  7.03e-02   |
      16  |  1.71e-02   |   4.1134    |  2.0403
      32  |  4.76e-03   |   3.5912    |  1.8445
      64  |  1.12e-03   |   4.2361    |  2.0827
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation 2D: Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.56e-01   |
      16  |  3.08e-02   |   5.0805    |  2.2001
      32  |  1.55e-02   |   1.9870    |  1.1651
      64  |  2.69e-03   |   5.7523    |  2.4072
    ---------------------------------------------
    Happy little convergence test!
    
    
    
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
       8  |  7.02e-02   |
      16  |  1.72e-02   |   4.0871    |  2.0311
      32  |  4.79e-03   |   3.5869    |  1.8427
      64  |  1.14e-03   |   4.2023    |  2.0712
    ---------------------------------------------
    Go Test Go!
    
    
    randomTensorMesh:  Interpolation 2D: Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.31e-01   |
      16  |  2.59e-02   |   5.0781    |  5.2443
      32  |  1.23e-02   |   2.0999    |  0.8534
      64  |  3.11e-03   |   3.9548    |  2.0421
    ---------------------------------------------
    You are awesome.
    
    
    
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
       8  |  7.05e-02   |
      16  |  1.77e-02   |   3.9862    |  1.9950
      32  |  4.41e-03   |   4.0153    |  2.0055
      64  |  1.17e-03   |   3.7677    |  1.9137
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation 2D: Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.96e-02   |
      16  |  1.73e-02   |   5.7675    |  2.0865
      32  |  1.01e-02   |   1.7044    |  1.2299
      64  |  2.77e-03   |   3.6586    |  1.7076
    ---------------------------------------------
    Happy little convergence test!
    
    
    
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
      16  |  1.90e-02   |   3.9961    |  1.9986
      32  |  4.59e-03   |   4.1347    |  2.0478
      64  |  1.20e-03   |   3.8324    |  1.9383
    ---------------------------------------------
    Testing is important.
    
    
    randomTensorMesh:  Interpolation 2D: Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  3.03e-01   |
      16  |  4.25e-02   |   7.1320    |  1.7548
      32  |  1.32e-02   |   3.2231    |  1.6324
      64  |  3.93e-03   |   3.3594    |  1.7875
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
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
       8  |  7.03e-02   |
      16  |  1.71e-02   |   4.1134    |  2.0403
      32  |  4.76e-03   |   3.5912    |  1.8445
      64  |  1.12e-03   |   4.2361    |  2.0827
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation 2D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.66e-01   |
      16  |  4.38e-02   |   6.0739    |  2.6888
      32  |  1.80e-02   |   2.4281    |  1.3972
      64  |  4.17e-03   |   4.3256    |  2.4146
    ---------------------------------------------
    Happy little convergence test!
    
    
    
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
       8  |  7.31e-02   |
      16  |  1.74e-02   |   4.2000    |  2.0704
      32  |  4.72e-03   |   3.6872    |  1.8825
      64  |  1.18e-03   |   4.0013    |  2.0005
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.08e-01   |
      16  |  4.56e-02   |   2.3716    |  0.8977
      32  |  1.24e-02   |   3.6634    |  2.2794
      64  |  1.88e-03   |   6.6126    |  2.7981
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
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
       8  |  6.98e-02   |
      16  |  1.88e-02   |   3.7045    |  1.8893
      32  |  4.59e-03   |   4.1091    |  2.0388
      64  |  1.14e-03   |   4.0151    |  2.0055
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  5.50e-02   |
      16  |  4.71e-02   |   1.1678    |  0.2570
      32  |  1.67e-02   |   2.8100    |  1.5992
      64  |  3.48e-03   |   4.8177    |  2.1148
    ---------------------------------------------
    Happy little convergence test!
    
    
    
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
      16  |  1.88e-02   |   3.7354    |  1.9013
      32  |  4.79e-03   |   3.9340    |  1.9760
      64  |  1.10e-03   |   4.3414    |  2.1181
    ---------------------------------------------
    And then everyone was happy.
    
    
    randomTensorMesh:  Interpolation Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.82e-01   |
      16  |  3.48e-02   |   5.2148    |  2.2201
      32  |  1.58e-02   |   2.2094    |  1.1758
      64  |  3.31e-03   |   4.7590    |  2.7468
    ---------------------------------------------
    Testing is important.
    
    
    
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
       8  |  7.03e-02   |
      16  |  1.88e-02   |   3.7298    |  1.8991
      32  |  4.71e-03   |   3.9968    |  1.9988
      64  |  1.18e-03   |   3.9906    |  1.9966
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation Ez
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  8.25e-02   |
      16  |  4.29e-02   |   1.9223    |  2.4876
      32  |  7.01e-03   |   6.1240    |  2.8538
      64  |  3.23e-03   |   2.1719    |  0.8988
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
       8  |  7.31e-02   |
      16  |  1.74e-02   |   4.2000    |  2.0704
      32  |  4.72e-03   |   3.6872    |  1.8825
      64  |  1.18e-03   |   4.0013    |  2.0005
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.20e-01   |
      16  |  3.87e-02   |   5.6726    |  1.9692
      32  |  8.15e-03   |   4.7505    |  2.5356
      64  |  2.92e-03   |   2.7923    |  1.1860
    ---------------------------------------------
    You are awesome.
    
    
    
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
      16  |  1.92e-02   |   3.9642    |  1.9870
      32  |  4.80e-03   |   4.0009    |  2.0003
      64  |  1.20e-03   |   4.0149    |  2.0054
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.28e-02   |
      16  |  4.98e-02   |   1.4609    |  0.5300
      32  |  1.33e-02   |   3.7498    |  2.0025
      64  |  3.39e-03   |   3.9258    |  2.0079
    ---------------------------------------------
    Go Test Go!
    
    
    
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
       8  |  7.00e-02   |
      16  |  1.77e-02   |   3.9486    |  1.9813
      32  |  4.72e-03   |   3.7558    |  1.9091
      64  |  1.18e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation Fz
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.23e-01   |
      16  |  2.42e-02   |   9.2112    |  3.4729
      32  |  1.01e-02   |   2.4067    |  1.3901
      64  |  2.60e-03   |   3.8724    |  1.5817
    ---------------------------------------------
    Go Test Go!
    
    
    
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
       8  |  6.98e-02   |
      16  |  1.88e-02   |   3.7045    |  1.8893
      32  |  4.59e-03   |   4.1091    |  2.0388
      64  |  1.14e-03   |   4.0151    |  2.0055
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.31e-01   |
      16  |  3.84e-02   |   3.3975    |  2.2016
      32  |  1.14e-02   |   3.3573    |  1.4353
      64  |  4.61e-03   |   2.4829    |  1.4794
    ---------------------------------------------
    You are awesome.
    
    
    
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
    Go Test Go!
    
    
    uniformLOM:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.42e-03   |
      32  |  6.06e-04   |   4.0001    |  2.0000
    ---------------------------------------------
    Happy little convergence test!
    
    
    rotateLOM:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.81e-03   |
      32  |  7.11e-04   |   3.9432    |  1.9794
    ---------------------------------------------
    Go Test Go!
    
    
    
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
    Happy little convergence test!
    
    
    uniformLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.29e-04   |
      32  |  1.57e-04   |   3.9978    |  1.9992
    ---------------------------------------------
    Testing is important.
    
    
    rotateLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.08e-04   |
      32  |  7.07e-05   |   4.3564    |  2.1231
    ---------------------------------------------
    Go Test Go!
    
    
    
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
    You get a gold star!
    
    
    uniformLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.99e-03   |
      32  |  1.75e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    The test be workin!
    
    
    rotateLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  7.70e-03   |
      32  |  1.94e-03   |   3.9622    |  1.9863
    ---------------------------------------------
    And then everyone was happy.
    
    
    
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
    Testing is important.
    
    
    uniformLOM:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.68e-03   |
      32  |  6.69e-04   |   3.9982    |  1.9993
    ---------------------------------------------
    Testing is important.
    
    
    rotateLOM:  Face Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.15e-03   |
      32  |  5.25e-04   |   4.0845    |  2.0302
    ---------------------------------------------
    That was easy!
    
    
    
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
    You are awesome.
    
    
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
    And then everyone was happy.
    
    
    
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
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.10e-03   |
      32  |  7.74e-04   |   3.9981    |  1.9993
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.54e-03   |
      32  |  6.23e-04   |   4.0741    |  2.0265
    ---------------------------------------------
    That was easy!
    
    
    
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
    You get a gold star!
    
    
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
    Go Test Go!
    
    
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
    And then everyone was happy.
    
    
    
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
    Testing is important.
    
    
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
    That was easy!
    
    
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
    Go Test Go!
    
    
    
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
    Once upon a time, a happy little test passed.
    
    
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
    You are awesome.
    
    
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
    Once upon a time, a happy little test passed.
    
    
    
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
    You are awesome.
    
    
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
    The test be workin!
    
    
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
    Happy little convergence test!
    
    
    
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
    Happy little convergence test!
    
    
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
    Happy little convergence test!
    
    
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
    You are awesome.
    
    
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
    Yay passed!
    
    
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
        <td><a href="javascript:showClassDetail('c11',1)">Detail</a></td>
    </tr>
    
    <tr id='pt11.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
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
    uniformTensorMesh:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Testing is important.
    
    
    uniformLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.71e-01   |
       8  |  8.86e-02   |   5.3170    |  2.4106
      16  |  1.70e-02   |   5.2040    |  2.3796
      32  |  3.77e-03   |   4.5126    |  2.1740
    ---------------------------------------------
    You get a gold star!
    
    
    
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
        <td><a href="javascript:showClassDetail('c12',1)">Detail</a></td>
    </tr>
    
    <tr id='pt12.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_order</div></td>
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
    uniformTensorMesh:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Testing is important.
    
    
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
    You are awesome.
    
    
    
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
    uniformTensorMesh:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    And then everyone was happy.
    
    
    uniformLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    Go Test Go!
    
    
    rotateLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.57e-03   |   3.6062    |  1.8505
    ---------------------------------------------
    You get a gold star!
    
    
    
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
    uniformTensorMesh:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    You are awesome.
    
    
    uniformLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    That was easy!
    
    
    rotateLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.57e-03   |
       8  |  6.54e-04   |   3.9234    |  1.9721
      16  |  1.80e-04   |   3.6283    |  1.8593
      32  |  4.66e-05   |   3.8703    |  1.9525
    ---------------------------------------------
    And then everyone was happy.
    
    
    
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
    uniformTensorMesh:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    Happy little convergence test!
    
    
    rotateLOM:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.56e-03   |
       8  |  6.54e-04   |   3.9232    |  1.9720
      16  |  1.80e-04   |   3.6343    |  1.8617
      32  |  4.64e-05   |   3.8804    |  1.9562
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr class='passClass'>
        <td>test_Solver.TestSolver</td>
        <td>10</td>
        <td>10</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c16',10)">Detail</a></td>
    </tr>
    
    <tr id='pt16.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.9' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt16.10' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.BasicTensorMeshTests</td>
        <td>7</td>
        <td>7</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c17',7)">Detail</a></td>
    </tr>
    
    <tr id='pt17.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_area_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorCC_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorN_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.TestPoissonEqn</td>
        <td>2</td>
        <td>2</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c18',2)">Detail</a></td>
    </tr>
    
    <tr id='pt18.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderBackward</div></td>
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
    uniformTensorMesh:  Poisson Equation - Backward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.22e-02   |
      20  |  7.96e-03   |   1.5342    |  1.9182
      24  |  5.59e-03   |   1.4258    |  1.9458
    ---------------------------------------------
    Testing is important.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt18.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderForward</div></td>
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
            
    pt18.2: 
    uniformTensorMesh:  Poisson Equation - Forward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.43e+00   |
      20  |  9.35e-01   |   1.5271    |  1.8974
      24  |  6.58e-01   |   1.4223    |  1.9320
    ---------------------------------------------
    Go Test Go!
    
    
    
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
        <td><a href="javascript:showClassDetail('c19',3)">Detail</a></td>
    </tr>
    
    <tr id='pt19.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFail</div></td>
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
            
    pt19.1: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.897e-01		5.751e-01		nan
    1	1.00e-02	2.862e-02		5.719e-02		1.002
    2	1.00e-03	2.857e-03		5.714e-03		1.000
    3	1.00e-04	2.857e-04		5.713e-04		1.000
    4	1.00e-05	2.857e-05		5.713e-05		1.000
    5	1.00e-06	2.857e-06		5.713e-06		1.000
    6	1.00e-07	2.857e-07		5.713e-07		1.000
    *********************************************************
    &lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt; FAIL! &gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;
    *********************************************************
    Coffee break?
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt19.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFunction</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt19.2')" >
            pass</a>
    
        <div id='div_pt19.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt19.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt19.2: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.519e-01		1.060e-02		nan
    1	1.00e-02	2.478e-02		1.129e-04		1.973
    2	1.00e-03	2.473e-03		1.137e-06		1.997
    3	1.00e-04	2.473e-04		1.137e-08		2.000
    4	1.00e-05	2.473e-05		1.137e-10		2.000
    5	1.00e-06	2.473e-06		1.137e-12		2.000
    6	1.00e-07	2.473e-07		1.144e-14		1.998
    ========================= PASS! =========================
    Yay passed!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt19.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simplePass</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt19.3')" >
            pass</a>
    
        <div id='div_pt19.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt19.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt19.3: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.580e-01		6.200e-03		nan
    1	1.00e-02	1.599e-02		6.068e-05		2.009
    2	1.00e-03	1.601e-03		6.055e-07		2.001
    3	1.00e-04	1.601e-04		6.054e-09		2.000
    4	1.00e-05	1.601e-05		6.054e-11		2.000
    5	1.00e-06	1.601e-06		6.053e-13		2.000
    6	1.00e-07	1.601e-07		6.020e-15		2.002
    ========================= PASS! =========================
    You get a gold star!
    
    
    
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
        <td><a href="javascript:showClassDetail('c20',8)">Detail</a></td>
    </tr>
    
    <tr id='pt20.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_invXXXBlockDiagonal</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc2</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc3</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt20.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='total_row'>
        <td>Total</td>
        <td>99</td>
        <td>99</td>
        <td>0</td>
        <td>0</td>
        <td>&nbsp;</td>
    </tr>
    </table>
    
