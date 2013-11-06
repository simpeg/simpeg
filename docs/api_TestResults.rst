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
    <p class='attribute'><strong>Start Time:</strong> 2013-11-05 16:10:07</p>
    <p class='attribute'><strong>Duration:</strong> 0:00:31.280082</p>
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
    0	1.00e-01	3.555e+01		3.750e+01		nan
    1	1.00e-02	1.711e-01		3.662e-01		2.010
    2	1.00e-03	1.586e-02		3.653e-03		2.001
    3	1.00e-04	1.915e-03		3.652e-05		2.000
    4	1.00e-05	1.948e-04		3.652e-07		2.000
    5	1.00e-06	1.951e-05		3.662e-09		1.999
    6	1.00e-07	1.951e-06		3.158e-11		2.064
    ========================= PASS! =========================
    Testing is important.
    
    
    
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
    0	1.00e-01	5.409e-02		4.593e-03		nan
    1	1.00e-02	5.429e-03		4.574e-05		2.002
    2	1.00e-03	5.431e-04		4.570e-07		2.000
    3	1.00e-04	5.431e-05		4.570e-09		2.000
    4	1.00e-05	5.431e-06		4.570e-11		2.000
    5	1.00e-06	5.431e-07		4.553e-13		2.002
    6	1.00e-07	5.431e-08		4.751e-15		1.982
    ========================= PASS! =========================
    That was easy!
    
    
    
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
    0	1.00e-01	8.756e+00		7.946e+00		nan
    1	1.00e-02	1.605e-01		7.946e-02		2.000
    2	1.00e-03	8.894e-03		7.946e-04		2.000
    3	1.00e-04	8.179e-04		7.946e-06		2.000
    4	1.00e-05	8.108e-05		7.946e-08		2.000
    5	1.00e-06	8.100e-06		7.946e-10		2.000
    6	1.00e-07	8.100e-07		7.947e-12		2.000
    ========================= PASS! =========================
    The test be workin!
    
    
    
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
    0	1.00e-01	1.309e-01		4.745e-17		nan
    1	1.00e-02	1.309e-02		4.825e-17		-0.007
    2	1.00e-03	1.309e-03		5.993e-17		-0.094
    3	1.00e-04	1.309e-04		6.648e-17		-0.045
    4	1.00e-05	1.309e-05		6.931e-17		-0.018
    5	1.00e-06	1.309e-06		6.247e-17		0.045
    6	1.00e-07	1.309e-07		5.708e-17		0.039
    ========================= PASS! =========================
    You are awesome.
    
    
    
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
    0	1.00e-01	3.133e-01		3.168e-02		nan
    1	1.00e-02	3.418e-02		3.168e-04		2.000
    2	1.00e-03	3.447e-03		3.168e-06		2.000
    3	1.00e-04	3.449e-04		3.168e-08		2.000
    4	1.00e-05	3.450e-05		3.168e-10		2.000
    5	1.00e-06	3.450e-06		3.169e-12		2.000
    6	1.00e-07	3.450e-07		3.481e-14		1.959
    ========================= PASS! =========================
    That was easy!
    
    
    
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
       8  |  2.76e-01   |
      16  |  6.10e-02   |   4.5221    |  2.1770
      32  |  1.44e-02   |   4.2290    |  2.0803
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation 1D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.46e-01   |
      16  |  1.11e-01   |   2.2066    |  0.9651
      32  |  4.07e-02   |   2.7360    |  3.1871
    ---------------------------------------------
    Testing is important.
    
    
    
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
       8  |  2.39e-01   |
      16  |  6.89e-02   |   3.4636    |  1.7923
      32  |  1.78e-02   |   3.8797    |  1.9559
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation 1D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  5.97e-01   |
      16  |  1.38e-01   |   4.3354    |  1.7681
      32  |  3.33e-02   |   4.1339    |  2.1579
    ---------------------------------------------
    Yay passed!
    
    
    
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
       8  |  7.06e-02   |
      16  |  1.78e-02   |   3.9781    |  1.9921
      32  |  4.72e-03   |   3.7595    |  1.9106
      64  |  1.18e-03   |   3.9981    |  1.9993
    ---------------------------------------------
    The test be workin!
    
    
    randomTensorMesh:  Interpolation 2D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.16e-02   |
      16  |  3.39e-02   |   2.7044    |  1.6128
      32  |  9.31e-03   |   3.6391    |  1.9612
      64  |  4.10e-03   |   2.2687    |  1.5130
    ---------------------------------------------
    Testing is important.
    
    
    
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
       8  |  7.04e-02   |
      16  |  1.88e-02   |   3.7329    |  1.9003
      32  |  4.77e-03   |   3.9517    |  1.9825
      64  |  1.19e-03   |   4.0081    |  2.0029
    ---------------------------------------------
    Testing is important.
    
    
    randomTensorMesh:  Interpolation 2D: Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.96e-01   |
      16  |  4.30e-02   |   4.5625    |  2.7996
      32  |  2.01e-02   |   2.1406    |  1.0860
      64  |  3.32e-03   |   6.0522    |  2.4217
    ---------------------------------------------
    You get a gold star!
    
    
    
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
       8  |  7.01e-02   |
      16  |  1.88e-02   |   3.7175    |  1.8943
      32  |  4.59e-03   |   4.1014    |  2.0361
      64  |  1.19e-03   |   3.8653    |  1.9506
    ---------------------------------------------
    The test be workin!
    
    
    randomTensorMesh:  Interpolation 2D: Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.92e-01   |
      16  |  5.13e-02   |   3.7457    |  2.6593
      32  |  1.09e-02   |   4.7164    |  2.0227
      64  |  4.07e-03   |   2.6707    |  1.4891
    ---------------------------------------------
    Go Test Go!
    
    
    
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
       8  |  7.06e-02   |
      16  |  1.78e-02   |   3.9781    |  1.9921
      32  |  4.72e-03   |   3.7595    |  1.9106
      64  |  1.18e-03   |   3.9981    |  1.9993
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation 2D: Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.82e-02   |
      16  |  5.58e-02   |   1.7592    |  6.6696
      32  |  6.57e-03   |   8.4887    |  2.2061
      64  |  4.28e-03   |   1.5342    |  0.6081
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
       8  |  7.57e-02   |
      16  |  1.88e-02   |   4.0222    |  2.0080
      32  |  4.72e-03   |   3.9905    |  1.9966
      64  |  1.18e-03   |   4.0110    |  2.0040
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 2D: Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  6.72e-02   |
      16  |  3.28e-02   |   2.0482    |  1.6433
      32  |  1.13e-02   |   2.9135    |  2.1786
      64  |  3.35e-03   |   3.3628    |  1.6438
    ---------------------------------------------
    Yay passed!
    
    
    
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
       8  |  7.04e-02   |
      16  |  1.88e-02   |   3.7329    |  1.9003
      32  |  4.77e-03   |   3.9517    |  1.9825
      64  |  1.19e-03   |   4.0081    |  2.0029
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation 2D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.24e-01   |
      16  |  6.30e-02   |   1.9636    |  1.2918
      32  |  2.15e-02   |   2.9310    |  1.6864
      64  |  4.96e-03   |   4.3359    |  1.9158
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
       8  |  7.36e-02   |
      16  |  1.74e-02   |   4.2291    |  2.0803
      32  |  4.16e-03   |   4.1900    |  2.0669
      64  |  9.48e-04   |   4.3858    |  2.1328
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.30e-01   |
      16  |  1.70e-02   |   13.5276    |  3.5152
      32  |  6.70e-03   |   2.5405    |  1.2651
      64  |  1.88e-03   |   3.5689    |  1.8184
    ---------------------------------------------
    That was easy!
    
    
    
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
      16  |  1.79e-02   |   3.8902    |  1.9598
      32  |  4.41e-03   |   4.0681    |  2.0244
      64  |  1.18e-03   |   3.7418    |  1.9037
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.92e-01   |
      16  |  8.20e-02   |   3.5597    |  4.3217
      32  |  9.49e-03   |   8.6460    |  1.9353
      64  |  2.89e-03   |   3.2854    |  1.7236
    ---------------------------------------------
    Testing is important.
    
    
    
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
      16  |  1.87e-02   |   3.7726    |  1.9155
      32  |  4.61e-03   |   4.0477    |  2.0171
      64  |  1.09e-03   |   4.2392    |  2.0838
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.30e-01   |
      16  |  4.52e-02   |   2.8770    |  1.4488
      32  |  1.36e-02   |   3.3196    |  2.2159
      64  |  2.81e-03   |   4.8481    |  1.8199
    ---------------------------------------------
    You are awesome.
    
    
    
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
      16  |  1.87e-02   |   3.7711    |  1.9150
      32  |  4.61e-03   |   4.0489    |  2.0175
      64  |  1.15e-03   |   4.0151    |  2.0054
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation Ez
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  3.81e-01   |
      16  |  3.89e-02   |   9.7974    |  2.2747
      32  |  2.09e-02   |   1.8614    |  2.2906
      64  |  3.97e-03   |   5.2691    |  1.7820
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
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
       8  |  7.36e-02   |
      16  |  1.74e-02   |   4.2291    |  2.0803
      32  |  4.16e-03   |   4.1900    |  2.0669
      64  |  9.48e-04   |   4.3858    |  2.1328
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.59e-01   |
      16  |  7.66e-02   |   2.0691    |  1.2662
      32  |  1.01e-02   |   7.5592    |  3.0198
      64  |  1.82e-03   |   5.5842    |  2.5778
    ---------------------------------------------
    That was easy!
    
    
    
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
      16  |  1.92e-02   |   3.9682    |  1.9885
      32  |  4.77e-03   |   4.0172    |  2.0062
      64  |  1.16e-03   |   4.1081    |  2.0385
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.45e-01   |
      16  |  5.01e-02   |   2.8858    |  1.2456
      32  |  8.71e-03   |   5.7525    |  3.4043
      64  |  1.85e-03   |   4.7102    |  1.9579
    ---------------------------------------------
    The test be workin!
    
    
    
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
       8  |  7.41e-02   |
      16  |  1.78e-02   |   4.1730    |  2.0611
      32  |  4.55e-03   |   3.9004    |  1.9636
      64  |  1.11e-03   |   4.0930    |  2.0332
    ---------------------------------------------
    And then everyone was happy.
    
    
    randomTensorMesh:  Interpolation Fz
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.63e-02   |
      16  |  4.19e-02   |   1.8225    |  1.1195
      32  |  9.07e-03   |   4.6176    |  1.4900
      64  |  2.70e-03   |   3.3657    |  2.0946
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
      16  |  1.79e-02   |   3.8902    |  1.9598
      32  |  4.41e-03   |   4.0681    |  2.0244
      64  |  1.18e-03   |   3.7418    |  1.9037
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.76e-01   |
      16  |  4.47e-02   |   6.1758    |  2.9429
      32  |  1.13e-02   |   3.9576    |  1.7499
      64  |  4.59e-03   |   2.4595    |  2.1993
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
    Once upon a time, a happy little test passed.
    
    
    uniformLOM:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.42e-03   |
      32  |  6.06e-04   |   4.0001    |  2.0000
    ---------------------------------------------
    Testing is important.
    
    
    rotateLOM:  Edge Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.81e-03   |
      32  |  7.11e-04   |   3.9432    |  1.9794
    ---------------------------------------------
    The test be workin!
    
    
    
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
    You are awesome.
    
    
    uniformLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.29e-04   |
      32  |  1.57e-04   |   3.9978    |  1.9992
    ---------------------------------------------
    Yay passed!
    
    
    rotateLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.08e-04   |
      32  |  7.07e-05   |   4.3564    |  2.1231
    ---------------------------------------------
    And then everyone was happy.
    
    
    
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
    Yay passed!
    
    
    uniformLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.99e-03   |
      32  |  1.75e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    Happy little convergence test!
    
    
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
    You are awesome.
    
    
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
    And then everyone was happy.
    
    
    uniformLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.79e-03   |
      32  |  1.70e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    You get a gold star!
    
    
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
    Testing is important.
    
    
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
    You are awesome.
    
    
    
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
    Go Test Go!
    
    
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
    And then everyone was happy.
    
    
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
    You get a gold star!
    
    
    
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
    And then everyone was happy.
    
    
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
    And then everyone was happy.
    
    
    
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
    You are awesome.
    
    
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
    And then everyone was happy.
    
    
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
    And then everyone was happy.
    
    
    
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
    And then everyone was happy.
    
    
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
    You get a gold star!
    
    
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
    Yay passed!
    
    
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
    Once upon a time, a happy little test passed.
    
    
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
    And then everyone was happy.
    
    
    
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
    You get a gold star!
    
    
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
    Testing is important.
    
    
    
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
    That was easy!
    
    
    uniformLOM:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.08e-03   |
      16  |  9.53e-04   |   9.5374    |  3.2536
      32  |  2.75e-04   |   3.4594    |  1.7905
    ---------------------------------------------
    That was easy!
    
    
    
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
    That was easy!
    
    
    uniformLOM:  Face Divergence 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.96e-01   |
      16  |  7.90e-02   |   3.7462    |  1.9054
      32  |  2.01e-02   |   3.9364    |  1.9769
      64  |  5.04e-03   |   3.9841    |  1.9943
    ---------------------------------------------
    The test be workin!
    
    
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
    Yay passed!
    
    
    uniformLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    You get a gold star!
    
    
    rotateLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.57e-03   |
       8  |  6.54e-04   |   3.9234    |  1.9721
      16  |  1.80e-04   |   3.6283    |  1.8593
      32  |  4.66e-05   |   3.8703    |  1.9525
    ---------------------------------------------
    You get a gold star!
    
    
    
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
    Go Test Go!
    
    
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
    Go Test Go!
    
    
    
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
    Go Test Go!
    
    
    
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
    Testing is important.
    
    
    
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
    0	1.00e-01	8.863e-02		1.778e-01		nan
    1	1.00e-02	8.915e-03		1.784e-02		0.999
    2	1.00e-03	8.921e-04		1.784e-03		1.000
    3	1.00e-04	8.921e-05		1.784e-04		1.000
    4	1.00e-05	8.921e-06		1.784e-05		1.000
    5	1.00e-06	8.921e-07		1.784e-06		1.000
    6	1.00e-07	8.921e-08		1.784e-07		1.000
    *********************************************************
    &lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt; FAIL! &gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;
    *********************************************************
    Testing is important. Do it again.
    
    
    
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
    0	1.00e-01	2.825e-01		1.846e-02		nan
    1	1.00e-02	2.803e-02		1.930e-04		1.981
    2	1.00e-03	2.799e-03		1.939e-06		1.998
    3	1.00e-04	2.799e-04		1.940e-08		2.000
    4	1.00e-05	2.799e-05		1.940e-10		2.000
    5	1.00e-06	2.799e-06		1.940e-12		2.000
    6	1.00e-07	2.799e-07		1.947e-14		1.998
    ========================= PASS! =========================
    The test be workin!
    
    
    
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
    0	1.00e-01	1.229e-01		5.540e-03		nan
    1	1.00e-02	1.249e-02		5.407e-05		2.011
    2	1.00e-03	1.251e-03		5.393e-07		2.001
    3	1.00e-04	1.251e-04		5.392e-09		2.000
    4	1.00e-05	1.251e-05		5.391e-11		2.000
    5	1.00e-06	1.251e-06		5.391e-13		2.000
    6	1.00e-07	1.251e-07		5.336e-15		2.004
    ========================= PASS! =========================
    Testing is important.
    
    
    
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
    
