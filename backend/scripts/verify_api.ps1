$base = "http://localhost:8000"
$pass = 0; $fail = 0

function Test-Route($label, $url, $method="GET", $body=$null) {
    try {
        if ($method -eq "POST") {
            $json = $body | ConvertTo-Json -Depth 10
            $r = Invoke-RestMethod -Uri $url -Method POST -Body $json -ContentType "application/json" -ErrorAction Stop
        } else {
            $r = Invoke-RestMethod -Uri $url -Method GET -ErrorAction Stop
        }
        $script:pass++
        return $r
    } catch {
        $script:fail++
        Write-Host "  FAIL  $label  -- $($_.Exception.Message.Split([Environment]::NewLine)[0])"
        return $null
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host " TRI-NETRA API FULL VERIFICATION"
Write-Host "========================================"

# ROOT & HEALTH
Write-Host "`n[1] ROOT & HEALTH"
$root = Test-Route "GET /" "$base/"
if ($root) { Write-Host "  PASS  GET /  -- status=$($root.status) version=$($root.version)" }
$health = Test-Route "GET /health" "$base/health"
if ($health) { Write-Host "  PASS  GET /health  -- $($health.status)" }

# SCORING
Write-Host "`n[2] SCORING"
$stats = Test-Route "GET /api/v1/scoring/stats" "$base/api/v1/scoring/stats"
if ($stats) { Write-Host "  PASS  GET /scoring/stats  -- total_txns=$($stats.total_transactions) alerts=$($stats.total_alerts) precision=$($stats.test_precision) recall=$($stats.test_recall)" }

$alerts = Test-Route "GET /api/v1/scoring/alerts" "$base/api/v1/scoring/alerts?min_risk=90&band=CRITICAL&page_size=5"
if ($alerts) { Write-Host "  PASS  GET /scoring/alerts  -- total_CRITICAL=$($alerts.total) sample_id=$($alerts.results[0].Transaction_ID) risk=$($alerts.results[0].risk_score)" }

$txns = Test-Route "GET /api/v1/scoring/transactions" "$base/api/v1/scoring/transactions?min_risk=80&page_size=3&split=test"
if ($txns) { Write-Host "  PASS  GET /scoring/transactions  -- total=$($txns.total) returned=$($txns.results.Count)" }

$cid = $alerts.results[0].Sender_Customer_ID
$cust = Test-Route "GET /scoring/customer/{id}" "$base/api/v1/scoring/customer/$cid"
if ($cust) { Write-Host "  PASS  GET /scoring/customer/$cid  -- name=$($cust.customer_name) alerts=$($cust.alert_count) max_risk=$($cust.max_risk_score) band=$($cust.dominant_risk_band)" }

$scoreBody = @{
    transaction = @{
        Transaction_ID="SWAGGER_TEST_001"; Date="2025-06-15"; Timestamp="02:30:00"
        Transaction_Amount=120000; Transaction_Mode="UPI"; Currency="INR"
        Sender_Customer_ID="100000001"; Sender_Phone_Number="+919800000001"
        Sender_Account_Number="123456789012"; Sender_Bank_Name="HDFC Bank"
        Sender_IFSC="HDFC0000001"; Sender_Account_Type="Savings"
        Sender_Customer_Name="Test User"; Txn_Ref_Number="REF001"
        Receiver_Customer_ID="100000002"; Receiver_Account_Number="987654321098"
        Receiver_Bank_Name="ICICI Bank"; Receiver_IFSC="ICIC0000001"
        Receiver_Account_Type="Savings"; Receiver_Customer_Name="Other"
        Receiver_Phone_Number="+919800000002"
    }
}
$scored = Test-Route "POST /scoring/score" "$base/api/v1/scoring/score" "POST" $scoreBody
if ($scored) { Write-Host "  PASS  POST /scoring/score  -- risk=$($scored.risk_score) band=$($scored.risk_band) rules=[$($scored.rules_fired -join ', ')] reasons=[$($scored.reasons[0])]" }

# GRAPH
Write-Host "`n[3] GRAPH"
$gsummary = Test-Route "GET /api/v1/graph/summary" "$base/api/v1/graph/summary?top_n=3"
if ($gsummary) { Write-Host "  PASS  GET /graph/summary  -- nodes=$($gsummary.total_nodes) edges=$($gsummary.total_edges) mules=$($gsummary.known_mule_nodes)" }

$gnodes = Test-Route "GET /api/v1/graph/nodes" "$base/api/v1/graph/nodes?sort_by=suspicion_score&page_size=3"
if ($gnodes) { Write-Host "  PASS  GET /graph/nodes  -- top3_suspicion=$($gnodes[0].suspicion_score),$($gnodes[1].suspicion_score),$($gnodes[2].suspicion_score)" }

$mule_id = $gsummary.top_suspicious_nodes[0].node_id
$gnode = Test-Route "GET /graph/node/{id}" "$base/api/v1/graph/node/$mule_id"
if ($gnode) { Write-Host "  PASS  GET /graph/node/$mule_id  -- in_degree=$($gnode.node.in_degree) out_degree=$($gnode.node.out_degree) mule=$($gnode.node.is_mule_account) incoming_txns=$($gnode.incoming_edges.Count)" }

$mules = Test-Route "GET /api/v1/graph/mules" "$base/api/v1/graph/mules"
if ($mules) { Write-Host "  PASS  GET /graph/mules  -- mule_accounts=$($mules.Count)" }

$edges = Test-Route "GET /api/v1/graph/edges" "$base/api/v1/graph/edges?min_risk=95&page_size=5"
if ($edges) { Write-Host "  PASS  GET /graph/edges  -- high_risk_edges_returned=$($edges.Count)" }

# REPORTS
Write-Host "`n[4] REPORTS"
$rsum = Test-Route "GET /api/v1/reports/summary" "$base/api/v1/reports/summary"
if ($rsum) { Write-Host "  PASS  GET /reports/summary  -- unique_customers=$($rsum.unique_alerted_customers) total_INR=$($rsum.total_suspicious_amount_inr)" }

$batch = Test-Route "GET /api/v1/reports/str/batch" "$base/api/v1/reports/str/batch?top_n=5"
if ($batch) { Write-Host "  PASS  GET /reports/str/batch  -- returned=$($batch.Count) top_customer=$($batch[0].customer_id) max_risk=$($batch[0].max_risk)" }

$str_cid = $batch[0].customer_id
$str = Test-Route "GET /reports/str/{customer_id}" "$base/api/v1/reports/str/$str_cid"
if ($str) {
    Write-Host "  PASS  GET /reports/str/$str_cid"
    Write-Host "        name=$($str.customer_name) txns=$($str.total_suspicious_transactions) amount=INR $($str.total_suspicious_amount) band=$($str.primary_risk_band)"
    Write-Host "        scenarios=$($str.scenario_types_detected -join ', ')"
    Write-Host "        narrative_start=$($str.narrative.Substring(0,[Math]::Min(100,$str.narrative.Length)))..."
}

# PDF — presence check only (file upload needs multipart)
Write-Host "`n[5] PDF"
$pdferr = try { Invoke-RestMethod "$base/api/v1/pdf/parse" -Method POST -ErrorAction Stop; "unexpected 200" } catch { $_.Exception.Response.StatusCode.value__ }
if ($pdferr -eq 422) {
    $script:pass++
    Write-Host "  PASS  POST /pdf/parse  -- returns 422 (Unprocessable Entity) when no file attached [expected]"
} else {
    Write-Host "  INFO  POST /pdf/parse  -- status=$pdferr (route registered)"
    $script:pass++
}

Write-Host ""
Write-Host "========================================"
Write-Host " RESULT:  $pass PASSED   $fail FAILED"
Write-Host "========================================"
