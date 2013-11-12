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
    <p class='attribute'><strong>Start Time:</strong> 2013-11-12 11:29:23</p>
    <p class='attribute'><strong>Duration:</strong> 0:00:32.452108</p>
    <p class='attribute'><strong>Status:</strong> Pass 107</p>
    
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
    0	1.00e-01	2.025e+01		2.686e+01		nan
    1	1.00e-02	3.953e-01		2.660e-01		2.004
    2	1.00e-03	6.347e-02		2.657e-03		2.000
    3	1.00e-04	6.587e-03		2.657e-05		2.000
    4	1.00e-05	6.610e-04		2.657e-07		2.000
    5	1.00e-06	6.613e-05		2.657e-09		2.000
    6	1.00e-07	6.613e-06		2.700e-11		1.993
    ========================= PASS! =========================
    You deserve a pat on the back!
    
    
    
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
    0	1.00e-01	5.087e-02		5.479e-03		nan
    1	1.00e-02	5.240e-03		5.471e-05		2.001
    2	1.00e-03	5.256e-04		5.468e-07		2.000
    3	1.00e-04	5.258e-05		5.467e-09		2.000
    4	1.00e-05	5.258e-06		5.467e-11		2.000
    5	1.00e-06	5.258e-07		5.467e-13		2.000
    6	1.00e-07	5.258e-08		5.672e-15		1.984
    ========================= PASS! =========================
    Testing is important.
    
    
    
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
    0	1.00e-01	6.461e+00		7.059e+00		nan
    1	1.00e-02	1.073e-02		7.059e-02		2.000
    2	1.00e-03	5.280e-03		7.059e-04		2.000
    3	1.00e-04	5.915e-04		7.059e-06		2.000
    4	1.00e-05	5.979e-05		7.059e-08		2.000
    5	1.00e-06	5.985e-06		7.059e-10		2.000
    6	1.00e-07	5.986e-07		7.059e-12		2.000
    ========================= PASS! =========================
    That was easy!
    
    
    
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
    0	1.00e-01	3.421e-01		3.925e-17		nan
    1	1.00e-02	3.421e-02		2.357e-17		0.222
    2	1.00e-03	3.421e-03		4.849e-17		-0.313
    3	1.00e-04	3.421e-04		5.925e-17		-0.087
    4	1.00e-05	3.421e-05		6.316e-17		-0.028
    5	1.00e-06	3.421e-06		4.061e-17		0.192
    6	1.00e-07	3.421e-07		6.370e-17		-0.196
    ========================= PASS! =========================
    You get a gold star!
    
    
    
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
    0	1.00e-01	2.565e+00		1.802e-01		nan
    1	1.00e-02	2.727e-01		1.802e-03		2.000
    2	1.00e-03	2.743e-02		1.802e-05		2.000
    3	1.00e-04	2.745e-03		1.802e-07		2.000
    4	1.00e-05	2.745e-04		1.802e-09		2.000
    5	1.00e-06	2.745e-05		1.802e-11		2.000
    6	1.00e-07	2.745e-06		1.800e-13		2.000
    ========================= PASS! =========================
    Yay passed!
    
    
    
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
       8  |  2.58e-01   |
      16  |  6.41e-02   |   4.0278    |  2.0100
      32  |  1.77e-02   |   3.6279    |  1.8591
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 1D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.97e-01   |
      16  |  1.59e-01   |   3.1240    |  1.6281
      32  |  4.04e-02   |   3.9430    |  2.3561
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
       8  |  3.09e-01   |
      16  |  7.89e-02   |   3.9199    |  1.9708
      32  |  1.83e-02   |   4.3119    |  2.1083
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation 1D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.76e-01   |
      16  |  1.11e-01   |   4.2662    |  1.9952
      32  |  4.59e-02   |   2.4283    |  1.2771
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    
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
       8  |  7.60e-02   |
      16  |  1.91e-02   |   3.9815    |  1.9933
      32  |  4.72e-03   |   4.0446    |  2.0160
      64  |  1.18e-03   |   4.0053    |  2.0019
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation 2D: CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.75e-01   |
      16  |  2.69e-02   |   6.5233    |  2.3005
      32  |  1.12e-02   |   2.4031    |  1.2481
      64  |  3.53e-03   |   3.1726    |  1.8392
    ---------------------------------------------
    You are awesome.
    
    
    
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
       8  |  7.01e-02   |
      16  |  1.88e-02   |   3.7205    |  1.8955
      32  |  4.59e-03   |   4.1006    |  2.0358
      64  |  1.16e-03   |   3.9440    |  1.9797
    ---------------------------------------------
    Well done Rowan!
    
    
    randomTensorMesh:  Interpolation 2D: Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.12e-01   |
      16  |  8.66e-02   |   1.2909    |  0.8416
      32  |  2.00e-02   |   4.3264    |  1.8545
      64  |  2.47e-03   |   8.1124    |  2.6759
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
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
       8  |  6.84e-02   |
      16  |  1.60e-02   |   4.2800    |  2.0976
      32  |  4.79e-03   |   3.3359    |  1.7381
      64  |  1.20e-03   |   3.9828    |  1.9938
    ---------------------------------------------
    The test be workin!
    
    
    randomTensorMesh:  Interpolation 2D: Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  6.00e-02   |
      16  |  4.16e-02   |   1.4412    |  0.4902
      32  |  7.60e-03   |   5.4787    |  2.8421
      64  |  2.60e-03   |   2.9195    |  1.2969
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
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
       8  |  7.60e-02   |
      16  |  1.91e-02   |   3.9815    |  1.9933
      32  |  4.72e-03   |   4.0446    |  2.0160
      64  |  1.18e-03   |   4.0053    |  2.0019
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation 2D: Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.56e-02   |
      16  |  3.64e-02   |   2.0777    |  0.8083
      32  |  1.07e-02   |   3.3892    |  1.8786
      64  |  3.58e-03   |   3.0004    |  1.6103
    ---------------------------------------------
    Go Test Go!
    
    
    
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
       8  |  7.49e-02   |
      16  |  1.80e-02   |   4.1557    |  2.0551
      32  |  4.00e-03   |   4.5049    |  2.1715
      64  |  1.20e-03   |   3.3419    |  1.7407
    ---------------------------------------------
    That was easy!
    
    
    randomTensorMesh:  Interpolation 2D: Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.96e-02   |
      16  |  5.79e-02   |   1.7215    |  0.5101
      32  |  1.21e-02   |   4.7877    |  2.1054
      64  |  2.60e-03   |   4.6488    |  2.4950
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
       8  |  7.01e-02   |
      16  |  1.88e-02   |   3.7205    |  1.8955
      32  |  4.59e-03   |   4.1006    |  2.0358
      64  |  1.16e-03   |   3.9440    |  1.9797
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation 2D: N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.13e-01   |
      16  |  4.02e-02   |   2.8137    |  1.7733
      32  |  1.12e-02   |   3.5726    |  1.4513
      64  |  3.25e-03   |   3.4586    |  1.4744
    ---------------------------------------------
    You get a gold star!
    
    
    
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
       8  |  7.59e-02   |
      16  |  1.90e-02   |   3.9963    |  1.9987
      32  |  4.69e-03   |   4.0474    |  2.0170
      64  |  1.18e-03   |   3.9935    |  1.9977
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    randomTensorMesh:  Interpolation CC
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.88e-02   |
      16  |  2.49e-02   |   3.1639    |  2.5664
      32  |  1.40e-02   |   1.7739    |  0.8582
      64  |  2.03e-03   |   6.9317    |  2.8790
    ---------------------------------------------
    And then everyone was happy.
    
    
    
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
       8  |  7.03e-02   |
      16  |  1.88e-02   |   3.7326    |  1.9002
      32  |  4.77e-03   |   3.9468    |  1.9807
      64  |  1.12e-03   |   4.2670    |  2.0932
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation Ex
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.15e-02   |
      16  |  7.21e-02   |   1.2700    |  0.5458
      32  |  1.22e-02   |   5.9140    |  2.6646
      64  |  2.75e-03   |   4.4260    |  1.8486
    ---------------------------------------------
    That was easy!
    
    
    
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
      16  |  1.88e-02   |   3.7324    |  1.9001
      32  |  4.32e-03   |   4.3577    |  2.1236
      64  |  1.19e-03   |   3.6424    |  1.8649
    ---------------------------------------------
    You get a gold star!
    
    
    randomTensorMesh:  Interpolation Ey
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.53e-01   |
      16  |  6.69e-02   |   2.2859    |  1.4178
      32  |  2.05e-02   |   3.2684    |  2.0771
      64  |  2.74e-03   |   7.4881    |  2.3296
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
      16  |  1.88e-02   |   3.7376    |  1.9021
      32  |  4.57e-03   |   4.1226    |  2.0435
      64  |  1.16e-03   |   3.9523    |  1.9827
    ---------------------------------------------
    You are awesome.
    
    
    randomTensorMesh:  Interpolation Ez
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  1.48e-01   |
      16  |  9.32e-02   |   1.5867    |  0.4554
      32  |  2.27e-02   |   4.1106    |  1.5265
      64  |  2.68e-03   |   8.4640    |  3.0885
    ---------------------------------------------
    Testing is important.
    
    
    
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
       8  |  7.59e-02   |
      16  |  1.90e-02   |   3.9963    |  1.9987
      32  |  4.69e-03   |   4.0474    |  2.0170
      64  |  1.18e-03   |   3.9935    |  1.9977
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    randomTensorMesh:  Interpolation Fx
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  8.49e-02   |
      16  |  2.72e-02   |   3.1225    |  1.2154
      32  |  1.12e-02   |   2.4354    |  1.9171
      64  |  3.67e-03   |   3.0455    |  1.3837
    ---------------------------------------------
    You deserve a pat on the back!
    
    
    
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
       8  |  7.60e-02   |
      16  |  1.91e-02   |   3.9765    |  1.9915
      32  |  4.72e-03   |   4.0514    |  2.0184
      64  |  1.18e-03   |   4.0055    |  2.0020
    ---------------------------------------------
    Happy little convergence test!
    
    
    randomTensorMesh:  Interpolation Fy
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  7.84e-02   |
      16  |  2.69e-02   |   2.9183    |  1.3569
      32  |  1.20e-02   |   2.2396    |  1.1828
      64  |  2.31e-03   |   5.1989    |  2.2465
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
       8  |  7.61e-02   |
      16  |  1.92e-02   |   3.9650    |  1.9873
      32  |  4.79e-03   |   4.0044    |  2.0016
      64  |  1.18e-03   |   4.0544    |  2.0195
    ---------------------------------------------
    Testing is important.
    
    
    randomTensorMesh:  Interpolation Fz
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  2.11e-01   |
      16  |  3.01e-02   |   7.0210    |  2.8286
      32  |  1.37e-02   |   2.1966    |  1.3516
      64  |  2.88e-03   |   4.7606    |  1.9822
    ---------------------------------------------
    Not just a pretty face Rowan
    
    
    
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
       8  |  7.03e-02   |
      16  |  1.88e-02   |   3.7326    |  1.9002
      32  |  4.77e-03   |   3.9468    |  1.9807
      64  |  1.12e-03   |   4.2670    |  2.0932
    ---------------------------------------------
    Yay passed!
    
    
    randomTensorMesh:  Interpolation N
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.98e-02   |
      16  |  3.64e-02   |   1.3674    |  0.3550
      32  |  1.19e-02   |   3.0611    |  2.4683
      64  |  3.98e-03   |   2.9868    |  1.4933
    ---------------------------------------------
    Yay passed!
    
    
    
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
    That was easy!
    
    
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
    Testing is important.
    
    
    
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
    Yay passed!
    
    
    uniformLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.29e-04   |
      32  |  1.57e-04   |   3.9978    |  1.9992
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Face Inner Product - Isotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.08e-04   |
      32  |  7.07e-05   |   4.3564    |  2.1231
    ---------------------------------------------
    The test be workin!
    
    
    
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
    Awesome, Rowan, just awesome.
    
    
    uniformLOM:  Edge Inner Product - Anisotropic
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.99e-03   |
      32  |  1.75e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    You deserve a pat on the back!
    
    
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
    Go Test Go!
    
    
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
    Not just a pretty face Rowan
    
    
    uniformLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  6.79e-03   |
      32  |  1.70e-03   |   3.9996    |  1.9998
    ---------------------------------------------
    Yay passed!
    
    
    rotateLOM:  Edge Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  7.49e-03   |
      32  |  1.89e-03   |   3.9617    |  1.9861
    ---------------------------------------------
    That was easy!
    
    
    
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
    That was easy!
    
    
    uniformLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  3.10e-03   |
      32  |  7.74e-04   |   3.9981    |  1.9993
    ---------------------------------------------
    Once upon a time, a happy little test passed.
    
    
    rotateLOM:  Face Inner Product - Full Tensor
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  2.54e-03   |
      32  |  6.23e-04   |   4.0741    |  2.0265
    ---------------------------------------------
    Go Test Go!
    
    
    
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
    Once upon a time, a happy little test passed.
    
    
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
    Yay passed!
    
    
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
    Not just a pretty face Rowan
    
    
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
    Happy little convergence test!
    
    
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
    Happy little convergence test!
    
    
    
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
    Yay passed!
    
    
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
    You get a gold star!
    
    
    
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
    You get a gold star!
    
    
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
    You get a gold star!
    
    
    
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
    The test be workin!
    
    
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
    That was easy!
    
    
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
    You are awesome.
    
    
    
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
    And then everyone was happy.
    
    
    uniformLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.43e-01   |
       8  |  1.48e-01   |   2.9914    |  1.5808
      16  |  3.95e-02   |   3.7462    |  1.9054
      32  |  1.00e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Happy little convergence test!
    
    
    rotateLOM:  Curl
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  4.71e-01   |
       8  |  8.86e-02   |   5.3170    |  2.4106
      16  |  1.70e-02   |   5.2040    |  2.3796
      32  |  3.77e-03   |   4.5126    |  2.1740
    ---------------------------------------------
    Go Test Go!
    
    
    
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
    Go Test Go!
    
    
    uniformLOM:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  4.44e-01   |
      16  |  1.19e-01   |   3.7462    |  1.9054
      32  |  3.01e-02   |   3.9364    |  1.9769
    ---------------------------------------------
    Well done Rowan!
    
    
    rotateLOM:  Face Divergence
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       8  |  9.08e-03   |
      16  |  9.53e-04   |   9.5374    |  3.2536
      32  |  2.75e-04   |   3.4594    |  1.7905
    ---------------------------------------------
    Testing is important.
    
    
    
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
    The test be workin!
    
    
    uniformLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    Happy little convergence test!
    
    
    rotateLOM:  Nodal Gradient
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.57e-03   |
       8  |  6.54e-04   |   3.9234    |  1.9721
      16  |  1.80e-04   |   3.6283    |  1.8593
      32  |  4.66e-05   |   3.8703    |  1.9525
    ---------------------------------------------
    The test be workin!
    
    
    
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
    Awesome, Rowan, just awesome.
    
    
    uniformLOM:  Nodal Gradient 2D
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
       4  |  2.00e-03   |
       8  |  5.25e-04   |   3.8065    |  1.9285
      16  |  1.34e-04   |   3.9116    |  1.9678
      32  |  3.39e-05   |   3.9578    |  1.9847
    ---------------------------------------------
    You get a gold star!
    
    
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
        <td><a href="javascript:showClassDetail('c16',4)">Detail</a></td>
    </tr>
    
    <tr id='pt16.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_GN_Rosenbrock</div></td>
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
            
    pt16.1: =========== Gauss Newton ===========
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
    
    <tr id='pt16.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_GN_quadratic</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt16.2')" >
            pass</a>
    
        <div id='div_pt16.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt16.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt16.2: =========== Gauss Newton ===========
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
    
    <tr id='pt16.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ProjGradient_quadratic1Bound</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt16.3')" >
            pass</a>
    
        <div id='div_pt16.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt16.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt16.3: ======================= Projected Gradient =======================
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
    
    <tr id='pt16.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ProjGradient_quadraticBounded</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt16.4')" >
            pass</a>
    
        <div id='div_pt16.4' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt16.4').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt16.4: ======================= Projected Gradient =======================
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
        <td><a href="javascript:showClassDetail('c17',14)">Detail</a></td>
    </tr>
    
    <tr id='pt17.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directDiagonal_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directFactored_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_1_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directLower_M_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.9' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.10' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directSpsolve_M</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.11' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.12' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_1_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.13' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M_fortran</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt17.14' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_directUpper_M_python</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.BasicTensorMeshTests</td>
        <td>7</td>
        <td>7</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c18',7)">Detail</a></td>
    </tr>
    
    <tr id='pt18.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_area_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt18.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt18.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_edge_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt18.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorCC_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt18.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vectorN_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt18.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt18.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_vol_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr class='passClass'>
        <td>test_tensorMesh.TestPoissonEqn</td>
        <td>2</td>
        <td>2</td>
        <td>0</td>
        <td>0</td>
        <td><a href="javascript:showClassDetail('c19',2)">Detail</a></td>
    </tr>
    
    <tr id='pt19.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderBackward</div></td>
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
    uniformTensorMesh:  Poisson Equation - Backward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.22e-02   |
      20  |  7.96e-03   |   1.5342    |  1.9182
      24  |  5.59e-03   |   1.4258    |  1.9458
    ---------------------------------------------
    You get a gold star!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt19.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_orderForward</div></td>
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
            
    pt19.2: 
    uniformTensorMesh:  Poisson Equation - Forward
    _____________________________________________
       h  |    error    | e(i-1)/e(i) |  order
    ~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~
      16  |  1.43e+00   |
      20  |  9.35e-01   |   1.5271    |  1.8974
      24  |  6.58e-01   |   1.4223    |  1.9320
    ---------------------------------------------
    You get a gold star!
    
    
    
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
        <td><a href="javascript:showClassDetail('c20',3)">Detail</a></td>
    </tr>
    
    <tr id='pt20.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFail</div></td>
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
            
    pt20.1: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	1.781e-01		3.576e-01		nan
    1	1.00e-02	1.794e-02		3.590e-02		0.998
    2	1.00e-03	1.795e-03		3.591e-03		1.000
    3	1.00e-04	1.795e-04		3.591e-04		1.000
    4	1.00e-05	1.795e-05		3.591e-05		1.000
    5	1.00e-06	1.795e-06		3.591e-06		1.000
    6	1.00e-07	1.795e-07		3.591e-07		1.000
    *********************************************************
    &lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt;&lt; FAIL! &gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;&gt;
    *********************************************************
    You break it, you fix it.
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt20.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simpleFunction</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt20.2')" >
            pass</a>
    
        <div id='div_pt20.2' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt20.2').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt20.2: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.266e-01		9.516e-03		nan
    1	1.00e-02	2.297e-02		9.227e-05		2.013
    2	1.00e-03	2.299e-03		9.201e-07		2.001
    3	1.00e-04	2.299e-04		9.199e-09		2.000
    4	1.00e-05	2.299e-05		9.198e-11		2.000
    5	1.00e-06	2.299e-06		9.198e-13		2.000
    6	1.00e-07	2.299e-07		9.155e-15		2.002
    ========================= PASS! =========================
    Well done Rowan!
    
    
    
            </pre>
        </div>
        <!--css div popup end-->
    
        </td>
    </tr>
    
    <tr id='pt20.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_simplePass</div></td>
        <td colspan='5' align='center'>
    
        <!--css div popup start-->
        <a class="popup_link" onfocus='this.blur();' href="javascript:showTestDetail('div_pt20.3')" >
            pass</a>
    
        <div id='div_pt20.3' class="popup_window">
            <div style='text-align: right; color:red;cursor:pointer'>
            <a onfocus='this.blur();' onclick="document.getElementById('div_pt20.3').style.display = 'none' " >
               [x]</a>
            </div>
            <pre>
            
    pt20.3: ==================== checkDerivative ====================
    iter	h		|J0-Jt|		|J0+h*dJ'*dx-Jt|	Order
    ---------------------------------------------------------
    0	1.00e-01	2.670e-01		1.197e-02		nan
    1	1.00e-02	2.767e-02		1.013e-04		2.072
    2	1.00e-03	2.775e-03		9.956e-07		2.008
    3	1.00e-04	2.776e-04		9.938e-09		2.001
    4	1.00e-05	2.776e-05		9.936e-11		2.000
    5	1.00e-06	2.776e-06		9.936e-13		2.000
    6	1.00e-07	2.776e-07		9.913e-15		2.001
    ========================= PASS! =========================
    You deserve a pat on the back!
    
    
    
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
        <td><a href="javascript:showClassDetail('c21',8)">Detail</a></td>
    </tr>
    
    <tr id='pt21.1' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.2' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_indexCube_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.3' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_invXXXBlockDiagonal</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.4' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc1</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.5' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc2</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.6' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_mkvc3</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.7' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_2D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='pt21.8' class='hiddenRow'>
        <td class='none'><div class='testcase'>test_ndgrid_3D</div></td>
        <td colspan='5' align='center'>pass</td>
    </tr>
    
    <tr id='total_row'>
        <td>Total</td>
        <td>107</td>
        <td>107</td>
        <td>0</td>
        <td>0</td>
        <td>&nbsp;</td>
    </tr>
    </table>
    
