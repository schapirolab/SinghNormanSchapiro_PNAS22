// Simulation 2 from Singh, Norman & Schapiro (2022)
// Article and additional information available at 10.1073/pnas.2123432119
// Information on how to run model is available at https://github.com/schapirolab/SinghNormanSchapiro_PNAS22

// Model developed in Emergent (www.github.com/emer/emergent)
// This simulation runs a hippocampal-cortical model on a continual learning task

package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"github.com/emer/emergent/patgen"

	"github.com/goki/ki/bitflag"

	"github.com/schapirolab/leabra-sleep/hip"
	"github.com/schapirolab/leabra-sleep/leabra"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net  *leabra.Network `view:"no-inline"`
	Pats *etable.Table   `view:"no-inline" desc:"the training patterns to use"` // ra25

	TrainAB *etable.Table `view:"no-inline" desc:"AB training patterns to use"`
	TrainAC *etable.Table `view:"no-inline" desc:"AC training patterns to use"`
	TestAB  *etable.Table `view:"no-inline" desc:"AB testing patterns to use"`
	TestAC  *etable.Table `view:"no-inline" desc:"AC testing patterns to use"`

	TrnTrlLog    *etable.Table     `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog    *etable.Table     `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats  *etable.Table     `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog    *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	TstStats     *etable.Table     `view:"no-inline" desc:"testing stats"`
	Params       params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag          string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params)"`
	MaxRuns      int               `desc:"maximum number of model runs to perform"`
	MaxEpcs      int               `desc:"maximum number of epochs to run per model run"`
	NZeroStop    int               `desc:"if a positive number, training will stop after this many epochs with zero mem errors"`
	TrialPerEpc  int               `desc:"number of trials per epoch of training"`
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`

	// StructSleep Implementation vars
	StrucSleepUpdt  leabra.TimeScales `desc:"at what time scale to update the display during strucsleep?  Anything longer than Epoch updates at Epoch in this model"`
	OscillStartCyc  int               `desc:"Structured sleep oscillation start cycle in minus phase -- 1 is default and means starting on the first minus phase cycle"`
	OscillStopCyc   int               `desc:"Structured sleep oscillation stop cycle in minus phase -- 75 is default and means stopping on the last minus phase cycle"`
	OscillAmplitude float64           `desc:"Structured sleep oscillation amplitude around midline"`
	OscillPeriod    float64           `desc:"Structured sleep oscillation period"`
	OscillMidline   float64           `desc:"Structured sleep oscillation midline - this is the value around which oscillation occurs`
	DispAvgEpcSSE   float64           `desc:"last test epoch's total sum squared error"`

	// Sleep implementation vars
	SleepEnv          env.FixedTable    `desc:"Training environment -- contains everything about iterating over sleep trials"`
	SlpCycLog         *etable.Table     `view:"no-inline" desc:"sleeping cycle-level log data"`
	SlpCycPlot        *eplot.Plot2D     `view:"-" desc:"the sleeping cycle plot"`
	MaxSlpCyc         int               `desc:"maximum number of cycle to sleep for a trial"`
	Sleep             bool              `desc:"Sleep or not"`
	LrnDrgSlp         bool              `desc:"Learning during sleep?"`
	SlpPlusThr        float32           `desc:"The threshold for entering a sleep plus phase"`
	SlpMinusThr       float32           `desc:"The threshold for entering a sleep minus phase"`
	InhibOscil        bool              `desc:"whether to implement inhibition oscillation"`
	SleepUpdt         leabra.TimeScales `desc:"at what time scale to update the display during sleep? Anything longer than Epoch updates at Epoch in this model"`
	InhibFactor       float64           `desc:"The inhib oscill factor for this cycle"`
	AvgLaySim         float64           `desc:"Average layer similaity between this cycle and last cycle"`
	SynDep            bool              `desc:"Syn Dep during sleep?"`
	SlpLearn          bool              `desc:"Learn during sleep?"`
	PlusPhase         bool              `desc:"Sleep Plusphase on/off"`
	MinusPhase        bool              `desc:"Sleep Minusphase on/off"`
	ZError            int               `desc:"Consec Zero error epochs"`
	ExecSleep         bool              `desc:"Execute Sleep?"`
	SlpTrls           int               `desc:"Number of sleep trials"`
	TstWrtOut         bool              `desc:"Write out Tst Acts? Set to false to reduce disk space consumption"`
	SlpPatMatchWrtOut bool              `desc:"Write out Sleep Pattern Decoding? Set to false to reduce disk space consumption"`

	// statistics: note use float64 as that is best for etable.Table
	TrlErr        float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"`
	TrlSSE        float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE     float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff    float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE        float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero SSE"`

	ABZero       bool     `inactive:"+" desc:"AB Testing Zero SSE"`
	ACZero       bool     `inactive:"+" desc:"AC Testing Zero SSE"`
	TrainABSSE   float64  `inactive:"+" desc:"AB Training SSE"`
	TrainACSSE   float64  `inactive:"+" desc:"AC Training SSE"`
	TestABSSE    float64  `inactive:"+" desc:"AB Testing SSE"`
	TestACSSE    float64  `inactive:"+" desc:"AC Testing SSE"`
	TestNm       string   `desc:"Which Test"`
	TstStatNms   []string `view:"-" desc:"Stats to split between AB, AC"`
	TestABCor    float64  `inactive:"+" desc:"AB Training Cor"` // For Sleep Thresh
	TestACCor    float64  `inactive:"+" desc:"AC Training Cor"`
	SleepStage   string   `inactive:"+" desc:"Stage of Sleep being run"`
	SWSCounter   int      `inactive:"+" desc:"Number of SWS blocks run"`
	REMCounter   int      `inactive:"+" desc:"Number of REM blocks run"`
	SleepCounter int      `inactive:"+" desc:"Number Sleep blocks run"`

	ClosestABA      int     `view:"-" desc:"Closest A"`
	ClosestABAMatch float32 `view:"-" desc:"Closest A Match %"`
	ClosestABB      int     `view:"-" desc:"Closest B"`
	ClosestABBMatch float32 `view:"-" desc:"Closest B Match %"`
	ClosestACA      int     `view:"-" desc:"Closest A''"`
	ClosestACAMatch float32 `view:"-" desc:"Closest A Match %"`
	ClosestACC      int     `view:"-" desc:"Closest C"`
	ClosestACCMatch float32 `view:"-" desc:"Closest B Match %"`

	// internal state - view:"-"
	SumErr       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnTrlPlot   *eplot.Plot2D               `view:"-" desc:"the training trial plot"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D               `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	TmpVals      []float32                   `view:"-" desc:"temp slice for holding values -- prevent mem allocs"`
	LayStatNms   []string                    `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	TstNms       []string                    `view:"-" desc:"names of test tables"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64                       `view:"-" desc:"the current random seed"`
	DirSeed      int64                       `view:"-" desc:"the current random seed for dir"`
	LastEpcTime  time.Time                   `view:"-" desc:"timer for last epoch"`
	ABover       int                         `view:"-" desc:"Overtrain counter AB"`
	ACover       int                         `view:"-" desc:"Overtrain counter AC"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.NewRndSeed()
	ss.MaxEpcs = 120
	ss.MaxRuns = 100
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{} // ra25
	ss.TrainAB = &etable.Table{}
	ss.TrainAC = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = SavedParamsSets
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.StrucSleepUpdt = leabra.AlphaCycle
	ss.TestInterval = 1
	ss.LogSetParams = false
	ss.LayStatNms = []string{"Input", "Output"}
	ss.TrialPerEpc = 10
	ss.TstWrtOut = false         // true to output tst trl acts
	ss.SlpPatMatchWrtOut = false // true to output sleep pattern deecoding

	ss.SlpCycLog = &etable.Table{}
	ss.Sleep = false
	ss.InhibOscil = true
	ss.SleepUpdt = leabra.Cycle
	ss.MaxSlpCyc = 50000
	ss.SynDep = true
	ss.SlpLearn = true
	ss.PlusPhase = false
	ss.MinusPhase = false
	ss.ExecSleep = true
	ss.SlpTrls = 0
	ss.OscillStartCyc = 1     // minus start cycle
	ss.OscillStopCyc = 75     // minus stop cycle
	ss.OscillAmplitude = 0.05 // amplitude around midline
	ss.OscillPeriod = 75.     // in cycles
	ss.OscillMidline = 1.     // horizontal zero value

	ss.SleepStage = "PreSleep"
	ss.SleepCounter = 0
	ss.SWSCounter = 0
	ss.REMCounter = 0

	ss.ABZero = false
	ss.TestNm = "AB"
	ss.TstNms = []string{"AB", "AC"}
	ss.TstStatNms = []string{"Err", "SSE", "AvgSSE"}

	ss.ABover = 0
	ss.ACover = 0

}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {

	ss.OpenPats()        // done
	ss.ConfigEnv()       // done except sleep
	ss.ConfigNet(ss.Net) // done
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)

	ss.ConfigSlpCycLog(ss.SlpCycLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
		//ss.NZeroStop = 5
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.SleepEnv.Nm = "SleepEnv"
	ss.SleepEnv.Dsc = "sleep params and state"
	ss.SleepEnv.Table = etable.NewIdxView(ss.Pats)
	ss.SleepEnv.Sequential = true
	ss.SleepEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
	ss.SleepEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {

	net.InitName(net, "sleep-replay-cortical")

	ext := net.AddLayer2D("EXT", 10, 12, emer.Input)
	inp := net.AddLayer2D("Input", 10, 12, emer.Hidden)
	out := net.AddLayer2D("Output", 10, 12, emer.Target)

	// Hipocampus!
	dg := net.AddLayer2D("DG", 15, 15, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", 12, 12, emer.Hidden)
	pca1 := net.AddLayer2D("pCA1", 10, 10, emer.Hidden)
	dca1 := net.AddLayer2D("dCA1", 10, 10, emer.Hidden)

	ctx := net.AddLayer2D("CTX", 20, 20, emer.Hidden)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	inp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "EXT", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "Input", YAlign: relpos.Front, Space: 2})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "DG", YAlign: relpos.Front, Space: 5})
	pca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 5})
	dca1.SetRelPos(relpos.Rel{Rel: relpos.FrontOf, Other: "pCA1", YAlign: relpos.Front, Space: 2})

	ctx.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "dCA1", YAlign: relpos.Back, Space: 5})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 5})

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	conn := prjn.NewFull()
	onetoone := prjn.NewOneToOne()

	spconn := prjn.NewUnifRnd()
	spconn.PCon = 0.6
	spconn.RndSeed = ss.RndSeed

	spconn2 := prjn.NewUnifRnd() // still needs testing
	spconn2.PCon = 0.1
	spconn2.RndSeed = ss.RndSeed

	spconn3 := prjn.NewUnifRnd()
	spconn3.PCon = 0.2
	spconn3.RndSeed = ss.RndSeed

	net.ConnectLayersPrjn(ext, inp, onetoone, emer.Forward, &hip.CHLPrjn{})
	net.ConnectLayersPrjn(out, inp, onetoone, emer.Back, &hip.CHLPrjn{})

	pj := net.ConnectLayersPrjn(inp, dg, spconn, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("PerDGPrjn")

	pj = net.ConnectLayersPrjn(dg, ca3, spconn2, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HipPrjn")

	pj = net.ConnectLayersPrjn(ca3, ca3, conn, emer.Lateral, &hip.CHLPrjn{})

	pj = net.ConnectLayersPrjn(ca3, pca1, spconn3, emer.Forward, &hip.CHLPrjn{})

	pj = net.ConnectLayersPrjn(pca1, out, conn, emer.Forward, &hip.CHLPrjn{})
	pj = net.ConnectLayersPrjn(out, pca1, conn, emer.Back, &hip.CHLPrjn{})

	pj = net.ConnectLayersPrjn(inp, dca1, conn, emer.Forward, &hip.CHLPrjn{})
	pj = net.ConnectLayersPrjn(dca1, out, conn, emer.Forward, &hip.CHLPrjn{})
	pj = net.ConnectLayersPrjn(out, dca1, conn, emer.Back, &hip.CHLPrjn{})

	pj = net.ConnectLayersPrjn(inp, ctx, conn, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("PerCTXPrjn")
	pj = net.ConnectLayersPrjn(ctx, out, conn, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("PerCTXPrjn")

	pj = net.ConnectLayersPrjn(inp, ca3, spconn3, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("PerDGPrjn")

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	dg.SetThread(1)
	ctx.SetThread(2)
	ca3.SetThread(3)
	pca1.SetThread(4)
	dca1.SetThread(5)

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView("train")
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
	//fmt.Println(ss.RndSeed)
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(state string) string { // changed from boolean to string
	if state == "train" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+
			" "+"%s\t TrialSSE:"+" "+"%.2f\t LastEpcSSE:"+" "+"%.2f\t\t\n TrainABSSE:"+" "+"%.2f\tTrainACSSE:"+
			" "+"%.2f\tTestABSSE:"+" "+"%.2f\tTestACSSE:"+" "+"%.2f\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur,
			ss.TrainEnv.Trial.Cur, ss.Time.Cycle, fmt.Sprintf(ss.TrainEnv.TrialName.Cur), ss.TrlSSE, ss.DispAvgEpcSSE,
			ss.TrainABSSE, ss.TrainACSSE, ss.TestABSSE, ss.TestACSSE)
	} else if state == "test" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+
			" "+"%s\t TrialSSE:"+" "+"%.2f\t LastEpcSSE:"+" "+"%.2f\t\t\n TrainABSSE:"+" "+"%.2f\tTrainACSSE:"+" "+
			"%.2f\tTestABSSE:"+" "+"%.2f\tTestACSSE:"+" "+"%.2f\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur,
			ss.TrainEnv.Trial.Cur, ss.Time.Cycle, fmt.Sprintf(ss.TrainEnv.TrialName.Cur), ss.TrlSSE, ss.DispAvgEpcSSE,
			ss.TrainABSSE, ss.TrainACSSE, ss.TestABSSE, ss.TestACSSE)
	} else if state == "sleep" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tCycle:"+" "+"%d\tInhibFactor: "+" "+
			"%.6f\tAvgLaySim: "+" "+"%.10f\t\t\t\nPlusPhase:"+" "+"%t\t MinusPhase:"+""+
			" "+"%t\t NearA:"+" "+"%d\tMatchA:"+" "+"%.2f\tNearB:"+" "+"%d\tMatchB:"+" "+"%.2f\t\t\t\nNearA':"+
			" "+"%d\tMatchA':"+" "+"%.2f\tNearC:"+" "+"%d\tMatchC:"+" "+"%.2f\tSlpTrls:"+" "+"%d\tSlpStage:"+" "+
			"%s\t\t\t\n",
			ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.Time.Cycle, ss.InhibFactor, ss.AvgLaySim, ss.PlusPhase,
			ss.MinusPhase, ss.ClosestABA, ss.ClosestABAMatch, ss.ClosestABB, ss.ClosestABBMatch, ss.ClosestACA,
			ss.ClosestACAMatch, ss.ClosestACC, ss.ClosestACCMatch, ss.SlpTrls, ss.SleepStage)

	} else if state == "strucsleep" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+" "+
			"%s\t InhibFactor: "+" "+"%.4f\t LastEpcSSE"+" "+"%.2f\t\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur,
			ss.SleepEnv.Trial.Cur, ss.Time.Cycle, fmt.Sprintf(ss.SleepEnv.TrialName.Cur), ss.InhibFactor, ss.DispAvgEpcSSE)
	}
	return ""
}

func (ss *Sim) UpdateView(state string) { // changed from boolean to string
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(state))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

// SleepCycInit prepares the network for spontaneous sleep
func (ss *Sim) SleepCycInit() {

	ss.Time.Reset()

	// Set all layers to be hidden
	for _, ly := range ss.Net.Layers {
		ly.SetType(emer.Hidden)
	}

	ss.Net.LayerByName("EXT").(leabra.LeabraLayer).AsLeabra().SetOff(true)
	ss.Net.InitActs()

	// Set all layers to random activation
	for _, ly := range ss.Net.Layers {
		for ni := range ly.(*leabra.Layer).Neurons {
			nrn := &ly.(*leabra.Layer).Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			msk := bitflag.Mask32(int(leabra.NeurHasExt))
			nrn.ClearMask(msk)
			rnd := rand.Float32()
			rnd = rnd - 0.5
			if rnd < 0 {
				rnd = 0
			}
			nrn.Act = rnd

		}
		ss.UpdateView("sleep")
	}

	// inc and dec set the rate at which synaptic depression increases and recovers at each synapse
	if ss.SynDep {
		for _, ly := range ss.Net.Layers {
			inc := 0.0009
			dec := 0.0005
			ly.(*leabra.Layer).InitSdEffWt(float32(inc), float32(dec))
		}
	}
}

// BackToWake terminates spontaneous sleep and sets the network up for wake training/testing again
func (ss *Sim) BackToWake() {
	// Effwt back to =Wt
	if ss.SynDep {
		for _, ly := range ss.Net.Layers {
			ly.(*leabra.Layer).TermSdEffWt()
		}
	}

	// Set the input/output/hidden layers back to normal.
	ss.Net.LayerByName("EXT").(leabra.LeabraLayer).AsLeabra().SetOff(false)

	ext := ss.Net.LayerByName("EXT").(leabra.LeabraLayer).AsLeabra()
	ext.SetType(emer.Input)

	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	inp.SetType(emer.Hidden)

	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	out.SetType(emer.Target)

}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	// Declare activation recording vars
	var ctxCycActs [][]float32

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time, false)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						if !train {
							ss.UpdateView("test")
						} else {
							ss.UpdateView("train")
						}

					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						if !train {
							ss.UpdateView("test")
						} else {
							ss.UpdateView("train")
						}
					}

				}
			}
			var ctxCycAct []float32

			ctx := ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra()
			ctx.UnitVals(&ctxCycAct, "Act")
			ctxCycActs = append(ctxCycActs, ctxCycAct)
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				if !train {
					ss.UpdateView("test")
				} else {
					ss.UpdateView("train")
				}
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					if !train {
						ss.UpdateView("test")
					} else {
						ss.UpdateView("train")
					}
				}
			}
		}
	}

	if !train && ss.TstWrtOut {
		dirpathacts := "output/" + "lrnacts" + "/" + "tstacts" + fmt.Sprint(ss.DirSeed) + "_truns_" + fmt.Sprint(ss.MaxRuns) + "/"

		if _, err := os.Stat(filepath.FromSlash(dirpathacts)); os.IsNotExist(err) {
			os.MkdirAll(filepath.FromSlash(dirpathacts), os.ModePerm)
		}

		filelrnacts, _ := os.OpenFile(filepath.FromSlash(dirpathacts+fmt.Sprint(ss.RndSeed)+"_"+"run"+fmt.Sprint(ss.TrainEnv.Run.Cur)+".csv"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		defer filelrnacts.Close()
		writerlrnacts := csv.NewWriter(filelrnacts)
		defer writerlrnacts.Flush()

		if (ss.TrainEnv.Epoch.Cur == 1) && (ss.TestEnv.TrialName.Cur == "evt_0_ab") {

			// copying params.go to better track params associated with the run data
			paramsdata, err := ioutil.ReadFile("params.go")
			if err != nil {
				fmt.Println(err)
				return
			}

			err = ioutil.WriteFile(filepath.FromSlash(dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"params.go"), paramsdata, 0644)
			if err != nil {
				fmt.Println("Error creating", dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"_"+"params.go")
				fmt.Println(err)
				return
			}

			mainfile, err := ioutil.ReadFile("simulation_2.go")
			if err != nil {
				fmt.Println(err)
				return
			}

			err = ioutil.WriteFile(dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"simulation_2.go", mainfile, 0644)
			if err != nil {
				fmt.Println("Error creating", dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"_"+"params.go")
				fmt.Println(err)
				return
			}

		}

		if (ss.TrainEnv.Epoch.Cur == 1) && (ss.TestEnv.TrialName.Cur == "evt_0_ab") {
			headers := []string{"Run", "Epoch", "Cycle", "TrialName", "SleepCounter"}

			for i := 0; i < 400; i++ {
				str := "CTX_" + fmt.Sprint(i)
				headers = append(headers, str)
			}
			if ss.TrainEnv.Epoch.Cur == 1 {
				writerlrnacts.Write(headers)
			}

		}
		valueStr := []string{}

		for i := 0; i < 100; i++ {
			if i == 19 || i == 99 {
				valueStr := []string{fmt.Sprint(ss.TrainEnv.Run.Cur), fmt.Sprint(ss.TrainEnv.Epoch.Cur), fmt.Sprint(i), fmt.Sprint(ss.TestEnv.TrialName.Cur), fmt.Sprint(ss.SleepCounter)}
				for _, vals := range ctxCycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				writerlrnacts.Write(valueStr)
			}
		}
		writerlrnacts.Write(valueStr)

	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {

		if !train {
			ss.UpdateView("test")
		} else {
			ss.UpdateView("train")
		}
	}
	if !train {
		//ss.UpdateView("test")
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end

	}

	if ss.TrainEnv.Run.Cur == 0 {
		ss.DirSeed = ss.RndSeed
	}

}

func (ss *Sim) StrucSleepAlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.StrucSleepUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}
	// Setting Output layer to type "output"
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	out.SetType(emer.Compare)

	// Storing current Gi values
	inpinhib := ss.Net.LayerByName("Input").(*leabra.Layer).Inhib.Layer.Gi
	dginhib := ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi
	ca3inhib := ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi
	ctxinhib := ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi
	outinhib := ss.Net.LayerByName("Output").(*leabra.Layer).Inhib.Layer.Gi

	pluscycs := 25

	inhiboscill := make([]float64, 0)

	// Padding for plus cycs + ss.OscillStartCyc
	for i := 0; i < pluscycs+ss.OscillStartCyc; i++ {
		inhiboscill = append(inhiboscill, 1.)
	}

	// Producing Oscill Gi factors
	for i := ss.OscillStartCyc; i <= ss.OscillStopCyc; i++ {
		inhiboscill = append(inhiboscill, ss.OscillAmplitude*math.Sin(2*3.14/ss.OscillPeriod*float64(i))+ss.OscillMidline)
	}

	// Padding after minus stop cycle
	for i := ss.OscillStopCyc; i < 75; i++ {
		inhiboscill = append(inhiboscill, 1.)
	}

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {

			ss.InhibFactor = inhiboscill[ss.Time.Cycle]
			// Changing Inhibs back to default before next oscill cycle value so that the inhib values are set based on inhib values
			ss.Net.LayerByName("Input").(*leabra.Layer).Inhib.Layer.Gi = inpinhib
			ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
			ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib
			ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi = ctxinhib
			ss.Net.LayerByName("Output").(*leabra.Layer).Inhib.Layer.Gi = outinhib

			oscilllayers := []string{"DG", "CA3", "CTX", "Output"} // layers to apply oscillating inhibition to
			for _, layer := range oscilllayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhiboscill[ss.Time.Cycle])
			}

			ss.Net.Cycle(&ss.Time, false) // Placed after Gi change

			// We only need the settled plus phase cycle but need all minus phase cycles
			if qtr == 0 && cyc == 24 {
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(true)
				}
			} else if qtr == 1 && cyc == 0 {
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(true)
				}
			} else if qtr > 0 && cyc > 0 {
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(false)
				}
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView("strucsleep")
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView("strucsleep")
					}
				}
			}
		}
		switch qtr {
		case 0:
			for _, ly := range ss.Net.Layers {
				ly.(leabra.LeabraLayer).AsLeabra().CalcActP(1)
			}
		case 3:
			for _, ly := range ss.Net.Layers {
				ly.(leabra.LeabraLayer).AsLeabra().CalcActM(75)
			}
		}

		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView("strucsleep")
			case viewUpdt == leabra.Phase:
				if qtr == 0 || qtr == 3 {
					ss.UpdateView("strucsleep")
				}
			}
		}
	}
	for _, lyc := range ss.Net.Layers {
		ly := ss.Net.LayerByName(lyc.Name()).(*leabra.Layer)
		for _, p := range ly.SndPrjns {
			if p.IsOff() {
				continue
			}
			p.(*hip.CHLPrjn).SlpDWt("err")
		}
	}
	out.SetType(emer.Target)
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView("strucsleep")
	}
	// Resetting Gis after sleep
	ss.Net.LayerByName("Input").(*leabra.Layer).Inhib.Layer.Gi = inpinhib
	ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
	ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib
	ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi = ctxinhib
	ss.Net.LayerByName("Output").(*leabra.Layer).Inhib.Layer.Gi = outinhib
}

func (ss *Sim) StrucSleepTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.SleepEnv.Step() // the Env encapsulates and manages all counter state

	ss.ApplyInputs(&ss.SleepEnv)
	ss.StrucSleepAlphaCyc(true) // train
}

func (ss *Sim) StrucSleepEpoch() {
	ss.SleepEnv.Run.Cur = ss.TrainEnv.Run.Cur
	ss.StopNow = false
	//curEpc := ss.TrainEnv.Epoch.Cur
	//curTrial := ss.TrainEnv.Trial.Cur
	for {
		ss.StrucSleepTrial()
		_, _, chg := ss.SleepEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"EXT", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	if ss.ABZero == false && ss.ACZero == false {
		ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
		ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(true)
		ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(true)
		ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
		ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
		ss.Net.GScaleFmAvgAct() // update computed scaling factors
		ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView("true")
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
			learnedAB := false

			if ss.TestABSSE == 0 && ss.TrainABSSE == 0 {
				ss.ABover++
			} else {
				ss.ABover = 0
			}

			if ss.ABover == 30 {
				learnedAB = true
			}

			//learnedAB := (ss.TestABSSE == 0 && ss.TrainABSSE == 0)
			if ss.TrainEnv.Table.Table == ss.TrainAB && learnedAB {
				ss.ABZero = true
				ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAC)

				// All Layers on. CTX learning rate lower.
				ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
				ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(false)
				ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(false)
				ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
				ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
				ss.Net.GScaleFmAvgAct() // update computed scaling factors
				ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

				ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().RcvPrjns.SendName("Input").(*hip.CHLPrjn).Learn.Lrate = 0.0001
				ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SndPrjns.RecvName("Output").(*hip.CHLPrjn).Learn.Lrate = 0.0001

			}

			//learnedAC := true
			learnedAC := (ss.TestACSSE == 0 && ss.TrainACSSE == 0)
			if ss.TrainEnv.Table.Table == ss.TrainAC && learnedAC {
				ss.ACZero = true

				if ss.ABZero == true && ss.ACZero == true {
					ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
					ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(true)
					ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(true)
					ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
					ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
					ss.Net.GScaleFmAvgAct() // update computed scaling factors
					ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
				}
				ss.TestAll()

				if ss.ABZero == true && ss.ACZero == true {
					ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
					ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(false)
					ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(false)
					ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
					ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
					ss.Net.GScaleFmAvgAct() // update computed scaling factors
					ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
				}

				for i := 0; i < 5; i++ {

					ss.InhibOscil = false
					ss.SleepStage = "SWS"
					if ss.SleepStage == "SWS" {
						ss.Net.LayerByName("DG").(*leabra.Layer).SetOff(false)
						ss.Net.LayerByName("CA3").(*leabra.Layer).SetOff(false)
						ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.GScaleFmAvgAct() // update computed scaling factors
						ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
						ss.SleepStage = "SWS"
						ss.SleepCounter += 1
						ss.SWSCounter += 1
					}
					cycles := 10000
					ss.SleepTrial("SWS", cycles)

					if ss.ABZero == true && ss.ACZero == true {
						ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.GScaleFmAvgAct() // update computed scaling factors
						ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
					}
					ss.TestAll()
					if ss.ABZero == true && ss.ACZero == true {
						ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.GScaleFmAvgAct() // update computed scaling factors
						ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
					}

					ss.InhibOscil = false
					ss.SleepStage = "REM"
					if ss.SleepStage == "REM" {
						ss.Net.LayerByName("DG").(*leabra.Layer).SetOff(true)
						ss.Net.LayerByName("CA3").(*leabra.Layer).SetOff(true)
						ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.GScaleFmAvgAct() // update computed scaling factors
						ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
						ss.SleepStage = "REM"
						ss.SleepCounter += 1
						ss.REMCounter += 1
					}

					ss.SleepTrial("REM", cycles)
					if ss.ABZero == true && ss.ACZero == true {
						ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(true)
						ss.Net.GScaleFmAvgAct() // update computed scaling factors
						ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
					}
					ss.TestAll()
					if ss.ABZero == true && ss.ACZero == true {
						ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra().SetOff(false)
						ss.Net.GScaleFmAvgAct() // update computed scaling factors
						ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
					}

				}

				ss.ABZero = false
				ss.ACZero = false
				ss.SleepCounter = 0
				ss.SleepStage = "PreSleep"
				ss.SlpTrls = 0
				learnedAB = false
				learnedAB = false

				ss.RunEnd()

				if ss.TrainEnv.Run.Incr() { // we are done!
					ss.StopNow = true
					return
				} else {
					ss.NeedsNewRun = true
					return
				}
			}
		}

		if epc >= ss.MaxEpcs { // ||  //(ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
}

// SleepCyc runs one 30,000 cycle trial of spontaneous sleep
func (ss *Sim) SleepCyc(c [][]float64, stage string, cycles int) {

	viewUpdt := ss.SleepUpdt

	stablecount := 0
	pluscount := 0
	minuscount := 0
	ss.SlpTrls = 0

	inp := ss.Net.LayerByName("Input").(*leabra.Layer)
	ctx := ss.Net.LayerByName("CTX").(*leabra.Layer)
	out := ss.Net.LayerByName("Output").(*leabra.Layer)
	ca3 := ss.Net.LayerByName("CA3").(*leabra.Layer)

	// Recording all inhibition Gi parameters prior to sleep for the inhibitory oscillations
	inpinhib := ss.Net.LayerByName("Input").(*leabra.Layer).Inhib.Layer.Gi
	dginhib := ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi
	ca3inhib := ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi
	ctxinhib := ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi
	outinhib := ss.Net.LayerByName("Output").(*leabra.Layer).Inhib.Layer.Gi
	pca1inhib := ss.Net.LayerByName("pCA1").(*leabra.Layer).Inhib.Layer.Gi
	dca1inhib := ss.Net.LayerByName("dCA1").(*leabra.Layer).Inhib.Layer.Gi

	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	writeout := [][]string{}

	// Loop for the 30,000 cycle sleep trial
	for cyc := 0; cyc < cycles; cyc++ { // 10000

		inp.SndPrjns.RecvName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.05
		out.RcvPrjns.SendName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.05

		inp.SndPrjns.RecvName("DG").(*hip.CHLPrjn).Learn.Learn = false
		inp.SndPrjns.RecvName("DG").(*hip.CHLPrjn).Learn.Learn = false
		ca3.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = false
		ca3.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn).Learn.Learn = false
		inp.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).Learn.Learn = false
		out.RcvPrjns.SendName("dCA1").(*hip.CHLPrjn).Learn.Learn = false
		out.RcvPrjns.SendName("pCA1").(*hip.CHLPrjn).Learn.Learn = false
		out.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn).Learn.Learn = false
		out.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).Learn.Learn = false

		ss.Net.WtFmDWt()

		ss.Net.Cycle(&ss.Time, true)
		ss.UpdateView("sleep")

		// Taking the prepared slice of oscil inhib values and producing the oscils in all perlys
		if ss.InhibOscil {
			inhibs := c                     // c is the slice with the sinwave values for the oscillating inhibition
			ss.InhibFactor = inhibs[0][cyc] // For sleep GUI counter and sleepcyclog

			// Changing Inhibs back to default before next oscill cycle value so that the inhib values are set based on c values
			ss.Net.LayerByName("Input").(*leabra.Layer).Inhib.Layer.Gi = inpinhib
			ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
			ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib
			ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi = ctxinhib
			ss.Net.LayerByName("Output").(*leabra.Layer).Inhib.Layer.Gi = outinhib
			ss.Net.LayerByName("pCA1").(*leabra.Layer).Inhib.Layer.Gi = pca1inhib
			ss.Net.LayerByName("dCA1").(*leabra.Layer).Inhib.Layer.Gi = dca1inhib

			// Two groups - low layers recieve lower-amplitude inhibitiory oscillations while high layers recive high-amplitude oscillations.
			// This is done to optimize oscillations for best minus-phases
			lowlayers := []string{"Input", "Output", "CTX", "pCA1", "dCA1"}
			highlayers := []string{"DG", "CA3"}

			for _, layer := range lowlayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhibs[0][cyc])
			}
			for _, layer := range highlayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhibs[1][cyc])
			}
		}

		// Average network similarity is the "stability" measure. It tracks the cycle-updated temporal auto-correlation of activation values at each layer.
		avesim := 0.0
		tmpsim := 0.0

		lys := []string{}
		if stage == "REM" {
			lys = []string{"Input", "Output", "CTX"}
		} else if stage == "SWS" {
			lys = []string{"Input", "Output", "CTX", "DG", "CA3", "pCA1", "dCA1"}
		}

		for _, lyc := range lys {
			ly := ss.Net.LayerByName(lyc).(*leabra.Layer)
			tmpsim = ly.Sim
			if math.IsNaN(tmpsim) {
				tmpsim = 0
			}
			actsum := float32(0)
			for ni := range ly.Neurons {
				nrn := &ly.Neurons[ni]
				act := nrn.Act
				actsum += act
			}
			if actsum < 1 {
				tmpsim = 0
			}
			avesim = avesim + tmpsim
		}
		ss.AvgLaySim = avesim / float64(len(lys))

		//If AvgLaySim falls below 0.9 - most likely because a layer has lost all act, random noise will be injected
		//into the network to get it going again. The first 1000 cycles are skipped to let the network initially settle into an attractor.
		if ss.Time.Cycle > 30000 && ss.AvgLaySim <= 0.8 && (ss.Time.Cycle%50 == 0 || ss.Time.Cycle%50 == 1 || ss.Time.Cycle%50 == 2 || ss.Time.Cycle%50 == 3 || ss.Time.Cycle%50 == 4) {
			for _, ly := range ss.Net.Layers {
				for ni := range ly.(*leabra.Layer).Neurons {
					nrn := &ly.(*leabra.Layer).Neurons[ni]
					if nrn.IsOff() {
						continue
					}
					nrn.Act = 0
					rnd := rand.Float32()
					rnd = rnd - 0.5
					if rnd < 0 {
						rnd = 0
					}
					nrn.Act = rnd
				}
			}
		}

		// Logging the SlpCycLog
		ss.LogSlpCyc(ss.SlpCycLog, ss.Time.Cycle)

		// Mark plus or minus phase
		if ss.SlpLearn {

			plusthresh := 0.9999
			minusthresh := plusthresh - 0.01

			if stage == "SWS" {
				plusthresh = 0.99995
				minusthresh = plusthresh - 0.0025
			} else if stage == "REM" {
				plusthresh = 0.999995
				minusthresh = plusthresh - 0.0025
			}

			// Checking if stable above threshold
			if ss.PlusPhase == false && ss.MinusPhase == false {
				if ss.AvgLaySim >= plusthresh {
					stablecount++
				} else if ss.AvgLaySim < plusthresh {
					stablecount = 0
				}
			}

			// For a dual threshold model, checking here if network has been stable above plusthresh for 5 cycles
			// Starting plus phase if criterion met
			if stablecount == 5 && ss.AvgLaySim >= plusthresh && ss.PlusPhase == false && ss.MinusPhase == false {
				stablecount = 0
				minuscount = 0
				ss.PlusPhase = true
				pluscount++
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(true)
				}

			} else if pluscount > 0 && ss.AvgLaySim >= plusthresh && ss.PlusPhase == true {
				pluscount++
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(false)
				}
				// If stabilty measure falls below plus threshold, plus phase ends and minus phase begins
			} else if ss.AvgLaySim < plusthresh && ss.AvgLaySim >= minusthresh && ss.PlusPhase == true {
				ss.PlusPhase = false
				ss.MinusPhase = true
				minuscount++

				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().CalcActP(pluscount)
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(true)
				}
				pluscount = 0

			} else if ss.AvgLaySim >= minusthresh && ss.MinusPhase == true {
				minuscount++
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().RunSumUpdt(false)
				}
				//	If stability measure falls below minus threshold, minus phase ends
			} else if ss.AvgLaySim < minusthresh && ss.MinusPhase == true {
				ss.MinusPhase = false

				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().CalcActM(minuscount)
				}
				minuscount = 0
				stablecount = 0

				for _, lyc := range ss.Net.Layers {

					ly := ss.Net.LayerByName(lyc.Name()).(*leabra.Layer)
					for _, p := range ly.SndPrjns {
						if p.IsOff() {
							continue
						}
						p.(*hip.CHLPrjn).SlpDWt("err") // Weight changes occuring here
					}
				}
				ss.SlpTrls++
				// Catching the rare occasion where stabilty drops in one cycle from above the plus threshold to below the minus threshold - ending trial if this happens
			} else if ss.AvgLaySim < minusthresh && ss.PlusPhase == true {
				ss.PlusPhase = false
				pluscount = 0
				stablecount = 0
				minuscount = 0
			}
		}

		// Forward the cycle timer
		ss.Time.CycleInc()

		ss.UpdateView("sleep")
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Cycle:
				ss.UpdateView("sleep")
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView("sleep")
				}
			case leabra.Quarter:
				if (cyc+1)%25 == 0 {
					ss.UpdateView("sleep")
				}
			case leabra.Phase:
				if (cyc+1)%100 == 0 {
					ss.UpdateView("sleep")
				}
			}
		}

		inp.SndPrjns.RecvName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.05
		out.RcvPrjns.SendName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.05

		inp.SndPrjns.RecvName("DG").(*hip.CHLPrjn).Learn.Learn = true
		inp.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = true
		ca3.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = true
		ca3.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn).Learn.Learn = true
		inp.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).Learn.Learn = true
		out.RcvPrjns.SendName("dCA1").(*hip.CHLPrjn).Learn.Learn = true
		out.RcvPrjns.SendName("pCA1").(*hip.CHLPrjn).Learn.Learn = true
		out.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn).Learn.Learn = true
		out.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).Learn.Learn = true

		var inpCycAct []float32
		inp.UnitVals(&inpCycAct, "Act")
		var outCycAct []float32
		out.UnitVals(&outCycAct, "Act")

		// NOTE: SatMatch will only return ONE of the Pats with the lowest errors. Multiple pats may have the same error but, it only returns first one
		minABA, minABB, ABAMatch, ABBMatch, minACA, minACC, ACAMatch, ACCMatch := ss.SatMatch(inpCycAct, outCycAct)

		writecyc := []string{}

		writecyc = append(writecyc, fmt.Sprint(ss.TrainEnv.Run.Cur), fmt.Sprint(ss.TrainEnv.Epoch.Cur),
			fmt.Sprint(ss.SleepCounter), fmt.Sprint(ss.PlusPhase), fmt.Sprint(ss.MinusPhase), fmt.Sprint(minABA),
			fmt.Sprint(ABAMatch), fmt.Sprint(minABB), fmt.Sprint(ABBMatch), fmt.Sprint(minACA), fmt.Sprint(ACAMatch),
			fmt.Sprint(minACC), fmt.Sprint(ACCMatch), fmt.Sprint(ss.SlpTrls))

		writeout = append(writeout, writecyc)

		outFmctx := ctx.SndPrjns.RecvName("Output").(*hip.CHLPrjn)

		var outFmctxeffwt []float32
		var outFmctxcai []float32
		var outFmctxsra []float32
		var outFmctxsdf []float32

		outFmctx.SynVals(&outFmctxeffwt, "Effwt")
		outFmctx.SynVals(&outFmctxcai, "Cai")
		outFmctx.SynVals(&outFmctxsra, "SenRecAct")
		outFmctx.SynVals(&outFmctxsdf, "SynDepFac")

	}

	dirpathacts := ""
	if ss.SlpPatMatchWrtOut {
		dirpathacts := "output/" + "sleep" + "/" + "ReplayMatch" + "/" + fmt.Sprint(ss.DirSeed) + "/" + "repmatch" +
			fmt.Sprint(ss.RndSeed) + "_truns_" + fmt.Sprint(ss.MaxRuns) + "_run_" + fmt.Sprint(ss.TrainEnv.Run.Cur) + "/"

		if _, err := os.Stat(filepath.FromSlash(dirpathacts)); os.IsNotExist(err) {
			os.MkdirAll(filepath.FromSlash(dirpathacts), os.ModePerm)
		}
	}

	if ss.SleepCounter == 1 && ss.SlpPatMatchWrtOut {
		// copying params.go to better track params associated with the run data
		paramsdata, err := ioutil.ReadFile("params.go")
		if err != nil {
			fmt.Println(err)
			return
		}

		err = ioutil.WriteFile(filepath.FromSlash("output/"+"sleep"+"/"+"ReplayMatch"+"/"+fmt.Sprint(ss.DirSeed)+
			"/"+"tstacts"+fmt.Sprint(ss.DirSeed)+"_"+"runs_"+fmt.Sprint(ss.MaxRuns)+"params.go"), paramsdata, 0644)
		if err != nil {
			fmt.Println("Error creating", dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"_"+"params.go")
			fmt.Println(err)
			return
		}

		mainfile, err := ioutil.ReadFile("simulation_2.go")
		if err != nil {
			fmt.Println(err)
			return
		}

		err = ioutil.WriteFile(filepath.FromSlash("output/"+"sleep"+"/"+"ReplayMatch"+"/"+fmt.Sprint(ss.DirSeed)+
			"/"+"tstacts"+fmt.Sprint(ss.DirSeed)+"_"+"runs_"+fmt.Sprint(ss.MaxRuns)+"simulation_2.go"),
			mainfile, 0644)
		if err != nil {
			fmt.Println("Error creating", dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"_"+"params.go")
			fmt.Println(err)
			return
		}
	}

	if ss.SlpPatMatchWrtOut {
		filew, _ := os.OpenFile(filepath.FromSlash(dirpathacts+"/"+"repmatch"+fmt.Sprint(ss.RndSeed)+"_"+
			"run"+fmt.Sprint(ss.TrainEnv.Run.Cur)+"epoch"+fmt.Sprint(ss.TrainEnv.Epoch.Cur)+
			"_"+"stage-"+fmt.Sprint(ss.SleepStage)+"slpblk_"+fmt.Sprint(ss.SleepCounter)+".csv"),
			os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

		defer filew.Close()
		writerw := csv.NewWriter(filew)
		defer writerw.Flush()

		headers := []string{"Run", "Epoch", "SlpCounter", "PlusPhase", "MinusPhase", "NearA", "AMatch",
			"NearB", "BMatch", "NearA'", "A'Match", "NearC", "CMatch", "SlpTrl"}
		writerw.Write(headers)
		writerw.Flush()

		writerw.WriteAll(writeout)
		writerw.Flush()
		filew.Close()
	}

	// Reset sleep algorithm variables
	pluscount = 0
	minuscount = 0
	ss.MinusPhase = false
	ss.PlusPhase = false
	stablecount = 0

	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	ss.Net.LayerByName("Input").(*leabra.Layer).Inhib.Layer.Gi = inpinhib
	ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
	ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib
	ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi = ctxinhib
	ss.Net.LayerByName("Output").(*leabra.Layer).Inhib.Layer.Gi = outinhib
	ss.Net.LayerByName("pCA1").(*leabra.Layer).Inhib.Layer.Gi = pca1inhib
	ss.Net.LayerByName("dCA1").(*leabra.Layer).Inhib.Layer.Gi = dca1inhib

	if ss.ViewOn {
		ss.UpdateView("sleep")
	}
}

// NOTE: SatMatch will only return ONE of the Pats with the lowest errors. Multiple pats may have the same error but it only returns first one
func (ss *Sim) SatMatch(inpact, outact []float32) (int, int, float32, float32, int, int, float32, float32) {

	file, err := os.Open("env1_pats_nohead.tsv")
	if err != nil {
		fmt.Println("err in reading file")
	}

	reader := csv.NewReader(file)
	reader.LazyQuotes = true
	reader.Comma = '\t'

	ABpatterns, err := reader.ReadAll()
	if err != nil {
		fmt.Println("err in reading reader object")
	}

	file, err = os.Open("env2_pats_nohead.tsv")
	if err != nil {
		fmt.Println("err in reading file")
	}

	reader = csv.NewReader(file)
	reader.LazyQuotes = true
	reader.Comma = '\t'

	ACpatterns, err := reader.ReadAll()
	if err != nil {
		fmt.Println("err in reading reader object")
	}

	inpbin := inpact
	outbin := outact

	ABAerrors := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	minABA := 0

	ABBerrors := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	minABB := 0

	for j := 0; j < 10; j++ {
		diff := []float64{}
		for i, val := range ABpatterns[j][:120] {
			valint, _ := strconv.Atoi(val)
			diff = append(diff, (math.Abs(float64(float32(valint) - inpbin[i]))))
		}
		for _, val := range diff {
			ABAerrors[j] += val
		}
		if j > 0 {
			if ABAerrors[j] < ABAerrors[minABA] {
				minABA = j
			}
		}
	}

	for j := 0; j < 10; j++ {
		diff := []float64{}
		for i, val := range ABpatterns[j][120:] {
			valint, _ := strconv.Atoi(val)
			diff = append(diff, (math.Abs(float64(float32(valint) - outbin[i]))))
		}
		for _, val := range diff {
			ABBerrors[j] += val
		}
		if j > 0 {
			if ABBerrors[j] < ABBerrors[minABB] {
				minABB = j
			}
		}
	}

	ACAerrors := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	minACA := 0

	ACCerrors := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	minACC := 0

	for j := 0; j < 10; j++ {
		diff := []float64{}
		for i, val := range ACpatterns[j][:120] {
			valint, _ := strconv.Atoi(val)
			diff = append(diff, (math.Abs(float64(float32(valint) - inpbin[i]))))
		}
		for _, val := range diff {
			ACAerrors[j] += val
		}
		if j > 0 {
			if ACAerrors[j] < ACAerrors[minACA] {
				minACA = j
			}
		}
	}

	for j := 0; j < 10; j++ {
		diff := []float64{}
		for i, val := range ACpatterns[j][120:] {
			valint, _ := strconv.Atoi(val)
			diff = append(diff, (math.Abs(float64(float32(valint) - outbin[i]))))
		}
		for _, val := range diff {
			ACCerrors[j] += val
		}
		if j > 0 {
			if ACCerrors[j] < ACCerrors[minACC] {
				minACC = j
			}
		}
	}

	ss.ClosestABA = minABA
	ss.ClosestABAMatch = float32(ABAerrors[minABA])

	ss.ClosestABB = minABB
	ss.ClosestABBMatch = float32(ABBerrors[minABB])

	ss.ClosestACA = minACA
	ss.ClosestACAMatch = float32(ACAerrors[minACA])

	ss.ClosestACC = minACC
	ss.ClosestACCMatch = float32(ACCerrors[minACC])

	return minABA, minABB, float32(ABAerrors[minABA]), float32(ABBerrors[minABB]),
		minACA, minACC, float32(ACAerrors[minACA]), float32(ACCerrors[minACC])

}

// SleepTrial sets up one spontaneous sleep trial
func (ss *Sim) SleepTrial(stage string, cycles int) {
	ss.SleepCycInit()
	ss.UpdateView("sleep")

	// Added for inhib oscill
	c := make([][]float64, 2)
	HighOscillAmp := 0.03 // 0.1 // 0.05
	LowOscillAmp := 0.06  // 0.07 // 0.0015 // 0.03
	OscillPeriod := 50.
	OscillMidline := 1.0

	// Generating Inhib Oscill Slices
	for i := 0; i < 100000; i++ {
		c[0] = append(c[0], LowOscillAmp*math.Sin(2*3.14/OscillPeriod*float64(i))+OscillMidline)  // low
		c[1] = append(c[1], HighOscillAmp*math.Sin(2*3.14/OscillPeriod*float64(i))+OscillMidline) // high
	}
	ss.SleepCyc(c, stage, cycles)
	ss.SlpCycPlot.GoUpdate()
	ss.BackToWake()
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {

	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {

	ss.ABZero = false
	ss.NewRndSeed()
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()

	dg := ss.Net.LayerByName("DG").(*leabra.Layer)
	ca3 := ss.Net.LayerByName("CA3").(*leabra.Layer)

	pjdgca3 := ca3.RcvPrjns.SendName("DG").(*hip.CHLPrjn)
	pjdgca3.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
	pjdgca3.Build()

	pjinpdg := dg.RcvPrjns.SendName("Input").(*hip.CHLPrjn)
	pjinpdg.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
	pjinpdg.Build()

	pjinpca3 := ca3.RcvPrjns.SendName("Input").(*hip.CHLPrjn)
	pjinpca3.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
	pjinpca3.Build()

	pjca3pca1 := ca3.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn)
	pjca3pca1.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
	pjca3pca1.Build()

	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false

	ss.Net.InitWts()

}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumErr = 0
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {

	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if ss.TrlSSE > 0 {
		ss.TrlErr = 1
	} else {
		ss.TrlErr = 0
	}
	if accum {
		ss.SumErr += ss.TrlErr
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
	}
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	curTrial := ss.TrainEnv.Trial.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc || curTrial == ss.TrialPerEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		vp.BlockUpdates()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.UnblockUpdates()
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView("test")
		}
		// Removed because of double counting issue in TstEpcLog
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accum
	ss.LogTstTrl(ss.TstTrlLog)
}

// TestItem tests given item which is at given index in test item list
// Currently Testitem will not do trialstats accum
func (ss *Sim) TestItem(idx int) {

	//outlay := ""
	//hide := ""

	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {

	ss.TestNm = "AB"
	ss.TestEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	ss.TestNm = "AC"
	ss.TestEnv.Table = etable.NewIdxView(ss.TrainAC)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true)
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("random_5x5_25_gen.csv", etable.Comma, etable.Headers)
}

func (ss *Sim) OpenPat(dt *etable.Table, fname, name, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func (ss *Sim) OpenPats() {
	ss.OpenPat(ss.TrainAB, "env1_pats.txt", "Env 1", "Env 1 Training Patterns")
	ss.OpenPat(ss.TrainAC, "env2_pats.txt", "Env 2", "Env 2 Training Patterns")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, (ss.TrainEnv.TrialName.Cur))
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("SSE", true, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	return plt
}

func (ss *Sim) LogSlpCyc(dt *etable.Table, cyc int) {

	row := dt.Rows
	if cyc == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	dt.SetCellFloat("InhibFactor", cyc, float64(ss.InhibFactor))
	dt.SetCellFloat("AvgLaySim", cyc, float64(ss.AvgLaySim))

	for _, ly := range ss.Net.Layers {
		lyc := ss.Net.LayerByName(ly.Name()).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name()+" Sim", row, float64(lyc.Sim))
	}

	ss.SlpCycPlot.GoUpdate()

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
	}
}

//DZ added
func (ss *Sim) ConfigSlpCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "SlpCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one sleep trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := ss.MaxSlpCyc // max cycles

	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"InhibFactor", etensor.FLOAT64, nil, nil},
		{"AvgLaySim", etensor.FLOAT64, nil, nil},
	}

	for _, ly := range ss.Net.Layers {
		sch = append(sch, etable.Column{ly.Name() + " Sim", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, np)
}

//DZ added
func (ss *Sim) ConfigSlpCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Sleep Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", true, true, 0, false, 0)
	plt.SetColParams("AvgLaySim", true, true, 0, true, 1)
	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv          // this is triggered by increment so use previous value
	nt := float64(len(ss.TrainEnv.Order)) // number of trials in view

	ss.EpcSSE = ss.SumSSE / nt
	ss.DispAvgEpcSSE = ss.EpcSSE

	if ss.TrainEnv.Table.Table == ss.TrainAB {
		ss.TrainABSSE = ss.EpcSSE
	}

	if ss.TrainEnv.Table.Table == ss.TrainAC {
		ss.TrainACSSE = ss.EpcSSE
	}

	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.SumErr) / nt
	ss.SumErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}

}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	trl := ss.TestEnv.Trial.Cur
	//row := trl
	//
	//if dt.Rows <= row {
	//	dt.SetNumRows(row + 1)
	//}

	row := dt.Rows
	if ss.TestNm == "AB" && trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	dt.SetCellString("TestNm", row, ss.TestNm)

	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	ivt := ss.ValsTsr("Input")
	ovt := ss.ValsTsr("Output")
	inp.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("InAct", row, ivt)
	out.UnitValsTensor(ovt, "ActM")
	dt.SetCellTensor("OutActM", row, ovt)
	out.UnitValsTensor(ovt, "ActP")
	dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestNm", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"InAct", etensor.FLOAT64, inp.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, out.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, out.Shp.Shp, nil},
	}...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TestNm", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
	}

	plt.SetColParams("InAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OutActM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OutActP", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv         // ?
	nt := float64(len(ss.TestEnv.Order)) // number of trials in view

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])
	dt.SetCellFloat("SlpTrls", row, float64(ss.SlpTrls))
	dt.SetCellString("PostSlpStg", row, ss.SleepStage)

	ss.DispAvgEpcSSE = agg.Sum(tix, "SSE")[0] / nt
	ss.UpdateView("test")

	// DS Added
	trix := etable.NewIdxView(trl)
	spl := split.GroupBy(trix, []string{"TestNm"})
	for _, ts := range ss.TstStatNms {
		split.Agg(spl, ts, agg.AggMean)
	}
	ss.TstStats = spl.AggsToTable(etable.ColNameOnly)

	for ri := 0; ri < ss.TstStats.Rows; ri++ {
		tst := ss.TstStats.CellString("TestNm", ri)
		for _, ts := range ss.TstStatNms {
			dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))
		}
	}

	// DS Added
	ss.TestABSSE = ss.TstStats.CellFloat("SSE", 0)
	ss.TestACSSE = ss.TstStats.CellFloat("SSE", 1)
	ss.TestABCor = 1 - ss.TstStats.CellFloat("Err", 0) // This is the AB Error
	ss.TestACCor = 1 - ss.TstStats.CellFloat("Err", 1) // This is the AC Error

	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("SSE", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "SSE", agg.AggSum)
	split.Agg(allsp, "AvgSSE", agg.AggMean)
	split.Agg(allsp, "InAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)

	ss.TstErrStats = allsp.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"SlpTrls", etensor.FLOAT64, nil, nil},
		{"PostSlpStg", etensor.STRING, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("AB SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AC SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", true, true, 0, true, .5)
		plt.SetColParams(lnm+" Act.Avg", true, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TstEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	//dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero)) // DS: Commente out to temporarily get rid of errors
	dt.SetCellFloat("ShSSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	for _, tn := range ss.TstNms {
		nm := tn + " " + "Mem"
		split.Desc(spl, nm)
	}
	split.Desc(spl, "FirstZero")
	ss.RunStats = spl.AggsToTable(false)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()

}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("SNS22_Sim2")
	gi.SetAppAbout(`This is a Sleep-replay model developed in Leabra. 
							See <a href="https://github.com/schapirolab/SinghNormanSchapiro_PNAS22"> on GitHub</a>.</p>`)

	win := gi.NewMainWindow("SNS22_Sim2", "SNS22_Sim2", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv
	nv.ViewDefaults()

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SlpCycPlot").(*eplot.Plot2D)
	ss.SlpCycPlot = ss.ConfigSlpCycPlot(plt, ss.SlpCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on trial -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, true, true) // contains, ignoreCase
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, true, false, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("sleep")

	tbar.AddAction(gi.ActOpts{Label: "Step StrucSleep Trial", Icon: "step-fwd", Tooltip: "Advances one structured sleep trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.StrucSleepTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step StrucSleep Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch of structured sleep trials at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.StrucSleepEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Spontaneous Sleep Trial", Icon: "fast-fwd", Tooltip: "Runs one spontaenous sleep trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.SleepTrial("SWS", 10000) //SWS for now but change later
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 9, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc" + strconv.Itoa(int(ss.RndSeed)))
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}
