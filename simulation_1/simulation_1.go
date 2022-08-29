// Simulation 1 from Singh, Norman & Schapiro (2022)
// Article and additional information available at 10.1073/pnas.2123432119
// Information on how to run model is available at https://github.com/schapirolab/SinghNormanSchapiro_PNAS22

// Model developed in Emergent (www.github.com/emer/emergent)
// This simulation runs a hippocampal-cortical model on the Satellite learning task (see Schapiro et al. (2017))

package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"github.com/goki/ki/bitflag"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"time"

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
	"github.com/goki/mat32"
)

func main() {
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
	Net          *leabra.Network   `view:"no-inline"`
	TrainSat     *etable.Table     `view:"no-inline" desc:"training patterns to use"`
	TestSat      *etable.Table     `view:"no-inline" desc:"testing patterns to use"`
	TrnTrlLog    *etable.Table     `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
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

	// DS: Sleep implementation vars
	SleepEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over sleep trials"`
	SlpCycLog    *etable.Table     `view:"no-inline" desc:"sleeping cycle-level log data"`
	SlpCycPlot   *eplot.Plot2D     `view:"-" desc:"the sleeping cycle plot"`
	MaxSlpCyc    int               `desc:"maximum number of cycle to sleep for a trial"`
	Sleep        bool              `desc:"Sleep or not"`
	LrnDrgSlp    bool              `desc:"Learning during sleep?"`
	SlpPlusThr   float32           `desc:"The threshold for entering a sleep plus phase"`
	SlpMinusThr  float32           `desc:"The threshold for entering a sleep minus phase"`
	InhibOscil   bool              `desc:"whether to implement inhibition oscillation"`
	SleepUpdt    leabra.TimeScales `desc:"at what time scale to update the display during sleep? Anything longer than Epoch updates at Epoch in this model"`
	InhibFactor  float64           `desc:"The inhib oscill factor for this cycle"`
	AvgLaySim    float64           `desc:"Average layer similaity between this cycle and last cycle"`
	SynDep       bool              `desc:"Syn Dep during sleep?"`
	SlpLearn     bool              `desc:"Learn during sleep?"`
	PlusPhase    bool              `desc:"Sleep Plusphase on/off"`
	MinusPhase   bool              `desc:"Sleep Minusphase on/off"`
	ZError       int               `desc:"Consec Zero error epochs"`
	ExecSleep    bool              `desc:"Execute Sleep?"`
	SlpTrls      int               `desc:"Number of sleep trials"`
	FinalTest    bool              `desc:"Flag for sleep occuring and this being the final test"`
	SlpTrlOcc    bool              `desc:"Bool to end sleep after first dwt to investigate each trial separately"`
	SlpWrtOut    bool              `desc:"Write out Sleep Acts? Set to false to reduce disk space consumption"`
	TstWrtOut    bool              `desc:"Write out Tst Acts? Set to false to reduce disk space consumption"`
	SlpTstWrtOut bool              `desc:"Write out Sleep Tst Epoch Acts? Set to false to reduce disk space consumption"`

	// statistics: note use float64 as that is best for etable.Table - DS Note: TrlSSE, TrlAvgSSE, TrlCosDiff don't need Shared and Unique vals... only accumulators do.
	TestNm     string  `inactive:"+" desc:"what set of patterns are we currently testing"`
	TrlSSE     float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`

	// These accumulators/Epc markers need separate Shared/Unique feature sums for tracking across epcs
	EpcShSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcShAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcShPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcShPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcShCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	ShFirstZero  int     `inactive:"+" desc:"epoch at when Mem err first went to zero"`
	ShNZero      int     `inactive:"+" desc:"number of epochs in a row with zero Mem err"`

	EpcUnSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcUnAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcUnPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcUnPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcUnCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	UnFirstZero  int     `inactive:"+" desc:"epoch at when Mem err first went to zero"`
	UnNZero      int     `inactive:"+" desc:"number of epochs in a row with zero Mem err"`

	// internal state - view:"-"
	// DS: Need separate Shared and Unique feature sums for tracking within epcs
	ShTrlNum     int     `inactive:"+" desc:"last epoch's total number of Shared Trials"`
	ShSumSSE     float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	ShSumAvgSSE  float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	ShSumCosDiff float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	ShCntErr     int     `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`

	UnTrlNum     int     `inactive:"+" desc:"last epoch's total number of Unique Trials"`
	UnSumSSE     float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	UnSumAvgSSE  float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	UnSumCosDiff float64 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	UnCntErr     int     `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`

	HiddenType    string `view:"-" inactive:"+" desc:"Feature type that is Hidden on this trial - Shared or Unique"`
	HiddenFeature string `view:"-" inactive:"+" desc:"Feature that is Hidden on this trial - F1-F5"`

	Win          *gi.Window       `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TrnTrlPlot   *eplot.Plot2D    `view:"-" desc:"the training trial plot"`
	TrnEpcPlot   *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D    `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File         `view:"-" desc:"log file"`
	RunFile      *os.File         `view:"-" desc:"log file"`
	TmpVals      []float32        `view:"-" desc:"temp slice for holding values -- prevent mem allocs"`
	LayStatNms   []string         `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	TstNms       []string         `view:"-" desc:"names of test tables"`
	SaveWts      bool             `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool             `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool             `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool             `view:"-" desc:"true if sim is running"`
	StopNow      bool             `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool             `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64            `view:"-" desc:"the current random seed"`
	DirSeed      int64            `view:"-" desc:"the current random seed for dir"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.NewRndSeed()
	ss.MaxEpcs = 30
	ss.Net = &leabra.Network{}
	ss.TrainSat = &etable.Table{}
	ss.TrainSat = &etable.Table{}
	ss.TestSat = &etable.Table{}
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
	ss.TestInterval = 1
	ss.LogSetParams = false
	ss.LayStatNms = []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName", "pCA1", "CTX", "DG"}
	ss.TstNms = []string{"Sat"}
	ss.TrialPerEpc = 50
	ss.ShTrlNum = 0
	ss.UnTrlNum = 0
	ss.MaxRuns = 100
	ss.ZError = 0

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
	ss.FinalTest = false
	ss.SlpTrlOcc = false
	ss.SlpWrtOut = false    // true to output sleep cyc acts
	ss.TstWrtOut = false    // true to output tst trl acts
	ss.SlpTstWrtOut = false // true to output extra test epoch results from both sides of sleep
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {

	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
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
		ss.NZeroStop = 1
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainSat)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Trial.Max = ss.TrialPerEpc
	ss.TrainEnv.Sequential = false

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestSat)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.SleepEnv.Nm = "SleepEnv"
	ss.SleepEnv.Dsc = "sleep params and state"
	ss.SleepEnv.Table = etable.NewIdxView(ss.TrainSat) // this is needed for the configenv to happen correctly even if no pats are ever shown
	ss.SleepEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
	ss.SleepEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "sleep-replay")

	// DS: Higer-level visual areas
	feature1 := net.AddLayer2D("F1", 6, 1, emer.Input)
	feature2 := net.AddLayer2D("F2", 6, 1, emer.Input)
	feature3 := net.AddLayer2D("F3", 6, 1, emer.Input)
	feature4 := net.AddLayer2D("F4", 6, 1, emer.Input)
	feature5 := net.AddLayer2D("F5", 6, 1, emer.Input)

	// DS: Higer-level language areas
	classname := net.AddLayer2D("ClassName", 1, 3, emer.Input)
	codename := net.AddLayer2D("CodeName", 6, 15, emer.Input)
	//
	//// DS: Hipocampus!
	dg := net.AddLayer2D("DG", 15, 15, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", 12, 12, emer.Hidden)
	pca1 := net.AddLayer2D("pCA1", 10, 10, emer.Hidden)
	dca1 := net.AddLayer2D("dCA1", 10, 10, emer.Hidden)

	// DS: Magic CTX
	ctx := net.AddLayer2D("CTX", 20, 20, emer.Hidden)

	feature1.SetClass("Per")
	feature2.SetClass("Per")
	feature3.SetClass("Per")
	feature4.SetClass("Per")
	feature5.SetClass("Per")
	//
	classname.SetClass("Per")
	codename.SetClass("Per")
	//
	dg.SetClass("Hip")
	ca3.SetClass("Hip")
	pca1.SetClass("Hip")
	dca1.SetClass("Hip")

	feature1.SetPos(mat32.Vec3{0, 20, 0})
	feature2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F1", YAlign: relpos.Front, Space: 2})
	feature3.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F2", YAlign: relpos.Front, Space: 2})
	feature4.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F3", YAlign: relpos.Front, Space: 2})
	feature5.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F4", YAlign: relpos.Front, Space: 2})
	codename.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "F5", YAlign: relpos.Front, Space: 2})
	classname.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "CodeName", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "F1", YAlign: relpos.Front, Space: 10})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "DG", YAlign: relpos.Front, Space: 5})
	pca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 5})
	dca1.SetRelPos(relpos.Rel{Rel: relpos.FrontOf, Other: "pCA1", YAlign: relpos.Front, Space: 2})
	ctx.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "dCA1", YAlign: relpos.Front, Space: 2})

	PerLays := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}

	conn := prjn.NewFull()

	spconn := prjn.NewUnifRnd()
	spconn.PCon = 0.6
	spconn.RndSeed = ss.RndSeed

	spconn2 := prjn.NewUnifRnd()
	spconn2.PCon = 0.1
	spconn2.RndSeed = ss.RndSeed

	spconn3 := prjn.NewUnifRnd()
	spconn3.PCon = 0.2
	spconn3.RndSeed = ss.RndSeed

	// Per-pCA1 / Per-Hip
	for _, lyc := range PerLays {

		ly := ss.Net.LayerByName(lyc).(leabra.LeabraLayer).AsLeabra()

		pj := net.ConnectLayersPrjn(ly, dg, spconn, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PerDGPrjn")

		pj = net.ConnectLayersPrjn(ly, ca3, spconn2, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PerCA3Prjn")

		pj = net.ConnectLayersPrjn(pca1, ly, conn, emer.Back, &hip.CHLPrjn{})
		pj.SetClass("PerCA1Prjn")

		pj = net.ConnectLayersPrjn(ly, dca1, conn, emer.Forward, &hip.CHLPrjn{}) //spconn3
		pj.SetClass("PerCA1Prjn")
		pj = net.ConnectLayersPrjn(dca1, ly, conn, emer.Back, &hip.CHLPrjn{})
		pj.SetClass("PerCA1Prjn")

		pj = net.ConnectLayersPrjn(ly, ctx, conn, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PerCTXPrjn")
		pj = net.ConnectLayersPrjn(ctx, ly, conn, emer.Back, &hip.CHLPrjn{})
		pj.SetClass("PerCTXPrjn")

		time.Sleep(1)
		ss.NewRndSeed()
	}

	pj := net.ConnectLayersPrjn(dg, ca3, spconn2, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HipPrjn")
	pj = net.ConnectLayersPrjn(ca3, ca3, conn, emer.Lateral, &hip.CHLPrjn{})
	pj = net.ConnectLayersPrjn(ca3, pca1, conn, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("PerCA1Prjn")

	//using 6 threads :)
	dg.SetThread(1)
	ctx.SetThread(2)
	ca3.SetThread(3)
	pca1.SetThread(4)
	dca1.SetThread(5)
	codename.SetThread(6)

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
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
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(state string) string { // changed from boolean to string
	if state == "train" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+
			" "+"%v\t\tHidden:"+" "+"%v\tFeature:"+" "+"%s\t\t\t\nShared Percent Correct:"+" "+
			"%.2f\t Unique Percent Correct"+" "+"%.2f\tUnique SSE"+" "+"%.2f\tShared SSE"+" "+
			"%.2f\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle,
			fmt.Sprintf(ss.TrainEnv.TrialName.Cur), ss.HiddenType, ss.HiddenFeature, ss.EpcShPctCor,
			ss.EpcUnPctCor, ss.EpcUnSSE, ss.EpcShSSE)
	} else if state == "test" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tTrial:"+" "+"%d\tCycle:"+" "+"%d\tName:"+" "+
			"%v\t\tHidden:"+" "+"%v\tFeature:"+" "+"%s\t\t\t\nShared Percent Correct:"+" "+
			"%.2f\t Unique Percent Correct"+" "+"%.2f\tUnique SSE"+" "+"%.2f\tShared SSE"+" "+
			"%.2f\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle,
			fmt.Sprintf(ss.TestEnv.TrialName.Cur), ss.HiddenType, ss.HiddenFeature, ss.EpcShPctCor,
			ss.EpcUnPctCor, ss.EpcUnSSE, ss.EpcShSSE)
	} else if state == "sleep" {
		return fmt.Sprintf("Run:"+" "+"%d\tEpoch:"+" "+"%d\tCycle:"+" "+"%d\tInhibFactor: "+" "+
			"%.6f\tAvgLaySim: "+" "+"%.6f\t\t\t\nShared Percent Correct:"+" "+"%.2f\t Unique Percent Correct:"+
			" "+"%.2f\t PlusPhase:"+" "+"%t\t MinusPhase:"+" "+"%t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur,
			ss.Time.Cycle, ss.InhibFactor, ss.AvgLaySim, ss.EpcShPctCor, ss.EpcUnPctCor, ss.PlusPhase, ss.MinusPhase)
	}
	return ""
}

func (ss *Sim) UpdateView(state string) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(state))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) SleepCycInit() {

	ss.Time.Reset()

	// Set all layers to be hidden
	for _, ly := range ss.Net.Layers {
		ly.SetType(emer.Hidden)
	}
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

	if ss.SynDep {
		for _, ly := range ss.Net.Layers {
			inc := 0.00035
			dec := 0.00025
			ly.(*leabra.Layer).InitSdEffWt(float32(inc), float32(dec))
		}
	}

}

func (ss *Sim) BackToWake() {
	// Effwt back to =Wt
	if ss.SynDep {
		for _, ly := range ss.Net.Layers {
			ly.(*leabra.Layer).TermSdEffWt()
		}
	}

	// Set the input/output/hidden layers back to normal.
	iolynms := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
	for _, lynm := range iolynms {
		ly := ss.Net.LayerByName(lynm).(leabra.LeabraLayer).AsLeabra()
		ly.SetType(emer.Input)
		ly.UpdateExtFlags()
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {

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
	var f1CycActs [][]float32
	var f2CycActs [][]float32
	var f3CycActs [][]float32
	var f4CycActs [][]float32
	var f5CycActs [][]float32
	var classCycActs [][]float32
	var codeCycActs [][]float32
	var pca1CycActs [][]float32
	var dca1CycActs [][]float32
	var ctxCycActs [][]float32
	var dgCycActs [][]float32
	var ca3CycActs [][]float32

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < 25; cyc++ {
			ss.Net.Cycle(&ss.Time, false)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView("train")
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView("train")
					}
				}
			}

			var f1CycAct []float32
			var f2CycAct []float32
			var f3CycAct []float32
			var f4CycAct []float32
			var f5CycAct []float32
			var classCycAct []float32
			var codeCycAct []float32
			var pca1CycAct []float32
			var dca1CycAct []float32
			var ctxCycAct []float32
			var dgCycAct []float32
			var ca3CycAct []float32

			f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
			f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
			f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
			f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
			f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
			classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
			codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()
			pca1 := ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra()
			dca1 := ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra()
			ctx := ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra()
			dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
			ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()

			f1.UnitVals(&f1CycAct, "Act")
			f1CycActs = append(f1CycActs, f1CycAct)
			f2.UnitVals(&f2CycAct, "Act")
			f2CycActs = append(f2CycActs, f2CycAct)
			f3.UnitVals(&f3CycAct, "Act")
			f3CycActs = append(f3CycActs, f3CycAct)
			f4.UnitVals(&f4CycAct, "Act")
			f4CycActs = append(f4CycActs, f4CycAct)
			f5.UnitVals(&f5CycAct, "Act")
			f5CycActs = append(f5CycActs, f5CycAct)
			classname.UnitVals(&classCycAct, "Act")
			classCycActs = append(classCycActs, classCycAct)
			codename.UnitVals(&codeCycAct, "Act")
			codeCycActs = append(codeCycActs, codeCycAct)

			pca1.UnitVals(&pca1CycAct, "Act")
			pca1CycActs = append(pca1CycActs, pca1CycAct)
			dca1.UnitVals(&dca1CycAct, "Act")
			dca1CycActs = append(dca1CycActs, dca1CycAct)
			ctx.UnitVals(&ctxCycAct, "Act")
			ctxCycActs = append(ctxCycActs, ctxCycAct)
			dg.UnitVals(&dgCycAct, "Act")
			dgCycActs = append(dgCycActs, dgCycAct)
			ca3.UnitVals(&ca3CycAct, "Act")
			ca3CycActs = append(ca3CycActs, ca3CycAct)

		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView("train")
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView("train")
				}
			}
		}
	}

	if ss.TrainEnv.Run.Cur == 0 {
		ss.DirSeed = ss.RndSeed
	}

	if !train && ss.TstWrtOut {
		dirpathacts := "output/" + "tst_acts/" + fmt.Sprint(ss.DirSeed) + "_truns_" +
			fmt.Sprint(ss.MaxRuns) + "_run_" + fmt.Sprint(ss.TrainEnv.Run.Cur)

		if _, err := os.Stat(filepath.FromSlash(dirpathacts)); os.IsNotExist(err) {
			os.MkdirAll(filepath.FromSlash(dirpathacts), os.ModePerm)
		}

		filelrnacts, _ := os.OpenFile(filepath.FromSlash(dirpathacts+"/"+"lrnacts"+fmt.Sprint(ss.RndSeed)+"_"+
			"run"+fmt.Sprint(ss.TrainEnv.Run.Cur)+"epoch"+fmt.Sprint(ss.TrainEnv.Epoch.Cur)+".csv"),
			os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

		defer filelrnacts.Close()
		writerlrnacts := csv.NewWriter(filelrnacts)
		defer writerlrnacts.Flush()

		if ss.TrainEnv.Epoch.Cur == 1 {

			// copying params.go to better track params associated with the run data
			paramsdata, err := ioutil.ReadFile("params.go")
			if err != nil {
				fmt.Println(err)
				return
			}

			err = ioutil.WriteFile(filepath.FromSlash(dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"params.go"),
				paramsdata, 0644)
			if err != nil {
				fmt.Println("Error creating", dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"_"+"params.go")
				fmt.Println(err)
				return
			}

			mainfile, err := ioutil.ReadFile("simulation_1.go")
			if err != nil {
				fmt.Println(err)
				return
			}

			err = ioutil.WriteFile(dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"simulation_1.go",
				mainfile, 0644)
			if err != nil {
				fmt.Println("Error creating", dirpathacts+"/"+fmt.Sprint(ss.DirSeed)+"_"+"params.go")
				fmt.Println(err)
				return
			}

		}

		if ss.TestEnv.Trial.Cur == 0 {
			headers := []string{"Run", "Epoch", "Cycle", "TrialName"}

			for i := 0; i < 6; i++ {
				str := "F1_" + fmt.Sprint(i)
				headers = append(headers, str)
			}
			for i := 0; i < 6; i++ {
				str := "F2_" + fmt.Sprint(i)
				headers = append(headers, str)
			}
			for i := 0; i < 6; i++ {
				str := "F3_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 6; i++ {
				str := "F4_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 6; i++ {
				str := "F5_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 3; i++ {
				str := "Class_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 90; i++ {
				str := "Code_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 225; i++ {
				str := "DG_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 400; i++ {
				str := "CTX_" + fmt.Sprint(i)
				headers = append(headers, str)
			}

			for i := 0; i < 100; i++ {
				str := "pCA1_" + fmt.Sprint(i)
				headers = append(headers, str)
			}
			for i := 0; i < 100; i++ {
				str := "dCA1_" + fmt.Sprint(i)
				headers = append(headers, str)
			}
			for i := 0; i < 144; i++ {
				str := "CA3_" + fmt.Sprint(i)
				headers = append(headers, str)
			}
			if !ss.FinalTest {
				writerlrnacts.Write(headers)
			}

		}
		valueStr := []string{}

		if ss.TrainEnv.Epoch.Cur == 10 {

		}

		for i := 0; i < 100; i++ {
			if i == 19 || i == 99 {
				valueStr := []string{fmt.Sprint(ss.TrainEnv.Run.Cur), fmt.Sprint(ss.TrainEnv.Epoch.Cur), fmt.Sprint(i),
					fmt.Sprint(ss.TestEnv.TrialName.Cur)}
				for _, vals := range f1CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range f2CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range f3CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range f4CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range f5CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range classCycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range codeCycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range dgCycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range ctxCycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range pca1CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range dca1CycActs[i] {
					valueStr = append(valueStr, fmt.Sprint(vals))
				}
				for _, vals := range ca3CycActs[i] {
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
		ss.UpdateView("train")
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}
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

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView("train")
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll(false)

			if ss.EpcShPctCor >= 0.66 && ss.EpcUnPctCor >= 0.66 {
				ss.TestAll(true) // Extra test right before sleep - results written to slp_tst dir

				if ss.ExecSleep && ss.SlpWrtOut {
					dirpathslp := "output/" + "slp_acts/" + fmt.Sprint(ss.DirSeed) + "/"
					if _, err := os.Stat(filepath.FromSlash(dirpathslp)); os.IsNotExist(err) {
						os.MkdirAll(filepath.FromSlash(dirpathslp), os.ModePerm)
					}
					fileslpres, _ := os.OpenFile(filepath.FromSlash(dirpathslp+"slpres.csv"),
						os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
					defer fileslpres.Close()
					writerslpres := csv.NewWriter(fileslpres)
					defer writerslpres.Flush()
					if ss.TrainEnv.Run.Cur == 0 {
						headers := []string{"Shared", "Unique", "ShSSE", "UnSSE", "SlpTrls"}
						writerslpres.Write(headers)
					}
					results := []string{strconv.FormatFloat(ss.EpcShPctCor, 'f', 6, 64),
						strconv.FormatFloat(ss.EpcUnPctCor, 'f', 6, 64),
						strconv.FormatFloat(ss.EpcShSSE, 'f', 6, 64),
						strconv.FormatFloat(ss.EpcUnSSE, 'f', 6, 64)}

					writerslpres.Write(results)
					//fmt.Println([]string{strconv.FormatFloat(ss.EpcShPctCor, 'f', 6, 64),
					//	strconv.FormatFloat(ss.EpcUnPctCor, 'f', 6, 64),
					//	strconv.FormatFloat(ss.EpcShSSE, 'f', 6, 64),
					//	strconv.FormatFloat(ss.EpcUnSSE, 'f', 6, 64)})

					ss.SleepTrial()
					ss.FinalTest = true
					//fmt.Println(ss.EpcShPctCor, ss.EpcUnPctCor, ss.EpcShSSE, ss.EpcUnSSE)
					ss.TestAll(true)
					results = []string{strconv.FormatFloat(ss.EpcShPctCor, 'f', 6, 64),
						strconv.FormatFloat(ss.EpcUnPctCor, 'f', 6, 64),
						strconv.FormatFloat(ss.EpcShSSE, 'f', 6, 64),
						strconv.FormatFloat(ss.EpcUnSSE, 'f', 6, 64), strconv.Itoa(ss.SlpTrls / 10)}

					writerslpres.Write(results)
					writerslpres.Flush()
					fileslpres.Close()
					ss.FinalTest = false
					//fmt.Println(ss.EpcShPctCor, ss.EpcUnPctCor, ss.EpcShSSE, ss.EpcUnSSE)
				}

				ss.RunEnd()
				if ss.TrainEnv.Run.Incr() {
					ss.StopNow = true
					return
				} else {
					ss.NeedsNewRun = true
					return
				}
			}

		}
		learned := ss.NZeroStop > 0 && ss.ShNZero >= ss.NZeroStop && ss.UnNZero >= ss.NZeroStop

		if learned || epc >= ss.MaxEpcs {
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() {
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	// Setting up train trial layer input/target chnages in this block
	f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
	f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
	f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
	f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
	f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
	classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
	codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()

	name := ss.TrainEnv.TrialName.Cur
	unique := 0
	shared := []string{"1", "2", "3", "4", "5", "classname"}
	r := rand.Float64()
	r1 := rand.Float64()
	outlay := ""

	for i, j := range name {
		if string(j) == "4" || string(j) == "5" || string(j) == "6" {
			unique = i + 1
			break
		}
	}

	for i, v := range shared {
		if (v) == strconv.Itoa(unique) {
			shared = append(shared[:i], shared[i+1:]...)
			break
		}
	}

	// Setting ratio for shared:unique feature hiding
	if r > 0.99 { // shared
		ss.HiddenType = "shared"
		hideindex := int(rand.Intn(len(shared)))
		ss.HiddenFeature = shared[hideindex]
		ss.ShTrlNum++
	} else { // unique
		if unique == 0 { // if there are no unique features, set codename to hide
			ss.HiddenType = "unique"
			ss.HiddenFeature = "codename"
			ss.UnTrlNum++
		} else {
			ss.HiddenType = "unique"
			ss.UnTrlNum++
			if r1 > 0.5 {
				ss.HiddenFeature = strconv.Itoa(unique)
			} else {
				ss.HiddenFeature = "codename"
			}
		}

	}

	switch ss.HiddenFeature {
	case "1":
		f1.SetType(emer.Target)
		f1.UpdateExtFlags()
		outlay = f1.Name()
	case "2":
		f2.SetType(emer.Target)
		f2.UpdateExtFlags()
		outlay = f2.Name()
	case "3":
		f3.SetType(emer.Target)
		f3.UpdateExtFlags()
		outlay = f3.Name()
	case "4":
		f4.SetType(emer.Target)
		f4.UpdateExtFlags()
		outlay = f4.Name()
	case "5":
		f5.SetType(emer.Target)
		f5.UpdateExtFlags()
		outlay = f5.Name()
	case "classname":
		classname.SetType(emer.Target)
		classname.UpdateExtFlags()
		outlay = classname.Name()
	case "codename":
		codename.SetType(emer.Target)
		codename.UpdateExtFlags()
		outlay = codename.Name()
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)

	ss.TrialStats(true, outlay)

	f1.SetType(emer.Input)
	f1.UpdateExtFlags()
	f2.SetType(emer.Input)
	f2.UpdateExtFlags()
	f3.SetType(emer.Input)
	f3.UpdateExtFlags()
	f4.SetType(emer.Input)
	f4.UpdateExtFlags()
	f5.SetType(emer.Input)
	f5.UpdateExtFlags()
	classname.SetType(emer.Input)
	classname.UpdateExtFlags()
	codename.SetType(emer.Input)
	codename.UpdateExtFlags()

	ss.LogTrnTrl(ss.TrnTrlLog)
}

func (ss *Sim) SleepCyc(c [][]float64) {

	viewUpdt := ss.SleepUpdt

	var f1CycActs [][]float32
	var f2CycActs [][]float32
	var f3CycActs [][]float32
	var f4CycActs [][]float32
	var f5CycActs [][]float32
	var classCycActs [][]float32
	var codeCycActs [][]float32

	var pca1CycActs [][]float32
	var dca1CycActs [][]float32
	var ctxCycActs [][]float32
	var dgCycActs [][]float32
	var ca3CycActs [][]float32

	var avglaysims []float32
	var inhibfacs []float32
	var plusphases []bool
	var minusphases []bool
	var pluscounts []int
	var minuscounts []int
	var stablecounts []int

	filetrnacts, _ := os.OpenFile("output/"+"slp_acts/"+fmt.Sprint(ss.DirSeed)+"/"+"acts"+
		fmt.Sprint(ss.RndSeed)+"_"+"run"+fmt.Sprint(ss.TrainEnv.Run.Cur)+"epoch"+fmt.Sprint(ss.TrainEnv.Epoch.Cur)+
		".csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)

	defer filetrnacts.Close()
	writertrnacts := csv.NewWriter(filetrnacts)
	defer writertrnacts.Flush()

	stablecount := 0
	pluscount := 0
	minuscount := 0
	ss.SlpTrls = 0

	// Getting Current Inhibs
	finhib := ss.Net.LayerByName("F1").(*leabra.Layer).Inhib.Layer.Gi
	clinhib := ss.Net.LayerByName("ClassName").(*leabra.Layer).Inhib.Layer.Gi
	coinhib := ss.Net.LayerByName("CodeName").(*leabra.Layer).Inhib.Layer.Gi
	dginhib := ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi
	ca3inhib := ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi
	ctxinhib := ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi
	pca1inhib := ss.Net.LayerByName("pCA1").(*leabra.Layer).Inhib.Layer.Gi
	dca1inhib := ss.Net.LayerByName("dCA1").(*leabra.Layer).Inhib.Layer.Gi

	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	pca1 := ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra()
	dca1 := ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra()

	perlys := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}
	for _, ly := range perlys {
		lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
		lyc.SndPrjns.RecvName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.03
		lyc.RcvPrjns.SendName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.03

		lyc.SndPrjns.RecvName("DG").(*hip.CHLPrjn).Learn.Learn = false
		lyc.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = false
		lyc.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).Learn.Learn = false
		lyc.RcvPrjns.SendName("pCA1").(*hip.CHLPrjn).Learn.Learn = false
		lyc.RcvPrjns.SendName("dCA1").(*hip.CHLPrjn).Learn.Learn = false

	}
	ca3.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = false
	ca3.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn).Learn.Learn = false

	dca1.SetOff(false)
	pca1.SetOff(false)

	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	for cyc := 0; cyc < 30000; cyc++ {
		ss.Net.WtFmDWt()

		ss.Net.Cycle(&ss.Time, true)
		ss.UpdateView("sleep")

		// Taking the prepared slice of oscil inhib values and producing the oscils in all
		// perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
		ss.InhibOscil = true
		if ss.InhibOscil {
			inhibs := c
			ss.InhibFactor = inhibs[0][cyc] // For sleep GUI counter and sleepcyclog

			// Changing Inhibs back to default before next oscill cycle value so that the inhib values follow a sinwave
			perlys := []string{"F1", "F2", "F3", "F4", "F5"}
			for _, ly := range perlys {
				ss.Net.LayerByName(ly).(*leabra.Layer).Inhib.Layer.Gi = finhib
			}
			ss.Net.LayerByName("ClassName").(*leabra.Layer).Inhib.Layer.Gi = clinhib
			ss.Net.LayerByName("CodeName").(*leabra.Layer).Inhib.Layer.Gi = coinhib
			ss.Net.LayerByName("pCA1").(*leabra.Layer).Inhib.Layer.Gi = pca1inhib
			ss.Net.LayerByName("dCA1").(*leabra.Layer).Inhib.Layer.Gi = dca1inhib
			ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
			ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi = ctxinhib
			ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib

			lowlayers := []string{"ClassName", "CTX", "pCA1", "dCA1"}
			highlayers := []string{"F1", "F2", "F3", "F4", "F5", "DG", "CA3"}

			for _, layer := range lowlayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhibs[0][cyc])
			}
			for _, layer := range highlayers {
				ly := ss.Net.LayerByName(layer).(*leabra.Layer)
				ly.Inhib.Layer.Gi = ly.Inhib.Layer.Gi * float32(inhibs[1][cyc])
			}
		}

		// Average network similarity
		avesim := 0.0
		tmpsim := 0.0
		for _, lyc := range ss.Net.Layers {
			ly := ss.Net.LayerByName(lyc.Name()).(*leabra.Layer)
			tmpsim = ly.Sim
			if math.IsNaN(tmpsim) {
				tmpsim = 0
			}
			avesim = avesim + tmpsim
		}
		ss.AvgLaySim = avesim / 12 // no. of lys

		// Logging the SlpCycLog
		ss.LogSlpCyc(ss.SlpCycLog, ss.Time.Cycle)

		// Mark plus or minus phase
		if ss.SlpLearn {
			plusthresh := 0.999965
			minusthresh := plusthresh - 0.0025

			// Checking if stable
			if ss.PlusPhase == false && ss.MinusPhase == false {
				if ss.AvgLaySim >= plusthresh {
					stablecount++
				} else if ss.AvgLaySim < plusthresh {
					stablecount = 0
				}
			}

			// For a dual threshold model, checking here if network has been stable above plusthresh for 5 cycles
			// Starting plus phase if criteria met
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
			} else if ss.AvgLaySim < plusthresh && ss.AvgLaySim >= minusthresh && ss.PlusPhase == true {

				ss.PlusPhase = false
				ss.MinusPhase = true
				minuscount++

				// Calculate final plusphase act avg for all synapses and store in syn var
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
			} else if ss.AvgLaySim < minusthresh && ss.MinusPhase == true {
				ss.MinusPhase = false

				// Calculate final minusphase act avg for all synapses and store in syn var
				for _, ly := range ss.Net.Layers {
					ly.(leabra.LeabraLayer).AsLeabra().CalcActM(minuscount)
				}
				minuscount = 0
				stablecount = 0

				//Dwt here
				if ss.SlpTrlOcc == false {
					for _, lyc := range ss.Net.Layers {
						ss.SlpTrls++
						ly := ss.Net.LayerByName(lyc.Name()).(*leabra.Layer)
						for _, p := range ly.SndPrjns {
							if p.IsOff() {
								continue
							}
							p.(*hip.CHLPrjn).SlpDWt("err")
						}
					}
				}

			} else if ss.AvgLaySim < minusthresh && ss.PlusPhase == true {
				ss.PlusPhase = false
				pluscount = 0
				stablecount = 0
				minuscount = 0
			}
		}

		if ss.SlpWrtOut {

			var f1CycAct []float32
			var f2CycAct []float32
			var f3CycAct []float32
			var f4CycAct []float32
			var f5CycAct []float32
			var classCycAct []float32
			var codeCycAct []float32

			var pca1CycAct []float32
			var dca1CycAct []float32
			var ctxCycAct []float32
			var dgCycAct []float32
			var ca3CycAct []float32

			var avglaysim float32
			var inhibfac float32
			var plusphase bool
			var minusphase bool
			var plcount int
			var micount int
			var stcount int

			f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
			f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
			f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
			f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
			f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
			classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
			codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()
			pca1 := ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra()
			dca1 := ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra()
			ctx := ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra()
			dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
			ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()

			f1.UnitVals(&f1CycAct, "Act")
			f1CycActs = append(f1CycActs, f1CycAct)
			f2.UnitVals(&f2CycAct, "Act")
			f2CycActs = append(f2CycActs, f2CycAct)
			f3.UnitVals(&f3CycAct, "Act")
			f3CycActs = append(f3CycActs, f3CycAct)
			f4.UnitVals(&f4CycAct, "Act")
			f4CycActs = append(f4CycActs, f4CycAct)
			f5.UnitVals(&f5CycAct, "Act")
			f5CycActs = append(f5CycActs, f5CycAct)
			classname.UnitVals(&classCycAct, "Act")
			classCycActs = append(classCycActs, classCycAct)
			codename.UnitVals(&codeCycAct, "Act")
			codeCycActs = append(codeCycActs, codeCycAct)

			pca1.UnitVals(&pca1CycAct, "Act")
			pca1CycActs = append(pca1CycActs, pca1CycAct)
			dca1.UnitVals(&dca1CycAct, "Act")
			dca1CycActs = append(dca1CycActs, dca1CycAct)
			ctx.UnitVals(&ctxCycAct, "Act")
			ctxCycActs = append(ctxCycActs, ctxCycAct)
			dg.UnitVals(&dgCycAct, "Act")
			dgCycActs = append(dgCycActs, dgCycAct)
			ca3.UnitVals(&ca3CycAct, "Act")
			ca3CycActs = append(ca3CycActs, ca3CycAct)

			avglaysim = float32(ss.AvgLaySim)
			avglaysims = append(avglaysims, avglaysim)

			inhibfac = float32(ss.InhibFactor)
			inhibfacs = append(inhibfacs, inhibfac)

			plusphase = ss.PlusPhase
			plusphases = append(plusphases, plusphase)
			minusphase = ss.MinusPhase
			minusphases = append(minusphases, minusphase)
			plcount = pluscount
			pluscounts = append(pluscounts, plcount)
			micount = minuscount
			minuscounts = append(minuscounts, micount)
			stcount = stablecount
			stablecounts = append(stablecounts, stcount)

			if ss.Time.Cycle == 0 {

				headers := []string{"AvgLaySim", "InhibFactor"}

				for i := 0; i < 6; i++ {
					str := "F1_" + fmt.Sprint(i)
					headers = append(headers, str)
				}
				for i := 0; i < 6; i++ {
					str := "F2_" + fmt.Sprint(i)
					headers = append(headers, str)
				}
				for i := 0; i < 6; i++ {
					str := "F3_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 6; i++ {
					str := "F4_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 6; i++ {
					str := "F5_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 3; i++ {
					str := "Class_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 90; i++ {
					str := "Code_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 225; i++ {
					str := "DG_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 400; i++ {
					str := "CTX_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				for i := 0; i < 100; i++ {
					str := "pCA1_" + fmt.Sprint(i)
					headers = append(headers, str)
				}
				for i := 0; i < 100; i++ {
					str := "dCA1_" + fmt.Sprint(i)
					headers = append(headers, str)
				}
				for i := 0; i < 144; i++ {
					str := "CA3_" + fmt.Sprint(i)
					headers = append(headers, str)
				}

				str := []string{"PlusPhase", "PlusCount", "MinusPhase", "MinusCount", "StableCount"}
				headers = append(headers, str...)

				writertrnacts.Write(headers)
			}
		}

		// Forward the cycle timer
		ss.Time.CycleInc()

		ss.UpdateView("sleep")
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Cycle:
				//			fmt.Scanln()
				ss.UpdateView("sleep")
			case leabra.FastSpike:
				if (cyc+1)%10 == 0 {
					ss.UpdateView("sleep")
					//ss.MonSlpCyc()
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
	}

	dca1.SetOff(false)
	pca1.SetOff(false)

	pluscount = 0
	minuscount = 0
	ss.MinusPhase = false
	ss.PlusPhase = false
	stablecount = 0

	if ss.SlpWrtOut {

		for i := 0; i < len(avglaysims); i++ {
			valueStr := []string{}

			valueStr = append(valueStr, fmt.Sprint(avglaysims[i]))
			valueStr = append(valueStr, fmt.Sprint(inhibfacs[i]))

			for _, vals := range f1CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range f2CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range f3CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range f4CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range f5CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range classCycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range codeCycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range dgCycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range ctxCycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range pca1CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range dca1CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}
			for _, vals := range ca3CycActs[i] {
				valueStr = append(valueStr, fmt.Sprint(vals))
			}

			valueStr = append(valueStr, fmt.Sprint(plusphases[i]))
			valueStr = append(valueStr, fmt.Sprint(pluscounts[i]))
			valueStr = append(valueStr, fmt.Sprint(minusphases[i]))
			valueStr = append(valueStr, fmt.Sprint(minuscounts[i]))
			valueStr = append(valueStr, fmt.Sprint(stablecounts[i]))

			writertrnacts.Write(valueStr)
		}
	}

	perlys = []string{"F1", "F2", "F3", "F4", "F5"}
	for _, ly := range perlys {
		ss.Net.LayerByName(ly).(*leabra.Layer).Inhib.Layer.Gi = finhib
	}

	ss.Net.LayerByName("ClassName").(*leabra.Layer).Inhib.Layer.Gi = clinhib
	ss.Net.LayerByName("CodeName").(*leabra.Layer).Inhib.Layer.Gi = coinhib
	ss.Net.LayerByName("pCA1").(*leabra.Layer).Inhib.Layer.Gi = pca1inhib
	ss.Net.LayerByName("dCA1").(*leabra.Layer).Inhib.Layer.Gi = dca1inhib
	ss.Net.LayerByName("DG").(*leabra.Layer).Inhib.Layer.Gi = dginhib
	ss.Net.LayerByName("CTX").(*leabra.Layer).Inhib.Layer.Gi = ctxinhib
	ss.Net.LayerByName("CA3").(*leabra.Layer).Inhib.Layer.Gi = ca3inhib

	perlys = []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}
	for _, ly := range perlys {
		lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
		lyc.SndPrjns.RecvName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.0001
		lyc.RcvPrjns.SendName("CTX").(*hip.CHLPrjn).Learn.Lrate = 0.0001
		lyc.SndPrjns.RecvName("DG").(*hip.CHLPrjn).Learn.Learn = true
		lyc.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = true
		lyc.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).Learn.Learn = true
		lyc.RcvPrjns.SendName("pCA1").(*hip.CHLPrjn).Learn.Learn = true
		lyc.RcvPrjns.SendName("dCA1").(*hip.CHLPrjn).Learn.Learn = true
	}

	ca3.SndPrjns.RecvName("CA3").(*hip.CHLPrjn).Learn.Learn = true
	ca3.SndPrjns.RecvName("pCA1").(*hip.CHLPrjn).Learn.Learn = true

	ss.Net.GScaleFmAvgAct() // update computed scaling factors
	ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	if ss.ViewOn {
		ss.UpdateView("sleep") // Update at the end of each sleep trials
	}
}

func (ss *Sim) SleepTrial() {
	ss.SleepCycInit()
	ss.UpdateView("sleep")

	// Added for inhib oscill
	c := make([][]float64, 2)
	HighOscillAmp := 0.05
	LowOscillAmp := 0.015
	OscillPeriod := 50.
	OscillMidline := 1.0

	for i := 0; i < 500000; i++ {
		c[0] = append(c[0], LowOscillAmp*math.Sin(2*3.14/OscillPeriod*float64(i))+OscillMidline)  // low
		c[1] = append(c[1], HighOscillAmp*math.Sin(2*3.14/OscillPeriod*float64(i))+OscillMidline) // high
	}

	ss.SleepCyc(c)
	ss.SlpCycPlot.GoUpdate() // make sure up-to-date at end
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
	ss.NewRndSeed()
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainSat)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()

	ss.InitStats()
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false

	dg := ss.Net.LayerByName("DG").(*leabra.Layer)
	ca3 := ss.Net.LayerByName("CA3").(*leabra.Layer)

	pjdgca3 := ca3.RcvPrjns.SendName("DG").(*hip.CHLPrjn)
	pjdgca3.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
	pjdgca3.Build()

	perlys := []string{"F1", "F2", "F3", "F4", "F5", "ClassName", "CodeName"}
	for _, layer := range perlys {
		time.Sleep(1)
		ss.NewRndSeed()

		pjperdg := dg.RcvPrjns.SendName(layer).(*hip.CHLPrjn)
		pjperdg.Pattern().(*prjn.UnifRnd).RndSeed = ss.RndSeed
		pjperdg.Build()
	}

	ss.Net.InitWts()

	ss.TrainEnv.Trial.Max = ss.TrialPerEpc

}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators for Shared Trials
	ss.ShSumSSE = 0
	ss.ShSumAvgSSE = 0
	ss.ShSumCosDiff = 0
	ss.ShCntErr = 0
	ss.ShFirstZero = -1
	ss.ShNZero = 0

	// accumulators for Unique Trials
	ss.UnSumSSE = 0
	ss.UnSumAvgSSE = 0
	ss.UnSumCosDiff = 0
	ss.UnCntErr = 0
	ss.UnFirstZero = -1
	ss.UnNZero = 0

	// epc tracking of shared/unique feature accums
	ss.EpcShSSE = 0
	ss.EpcShAvgSSE = 0
	ss.EpcShPctErr = 0
	ss.EpcShCosDiff = 0

	ss.EpcUnSSE = 0
	ss.EpcUnAvgSSE = 0
	ss.EpcUnPctErr = 0
	ss.EpcUnCosDiff = 0

}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool, outlaynm string) (sse, avgsse, cosdiff float64) {

	// Getting outlay
	//fmt.Println(outlaynm)
	outLay := ss.Net.LayerByName(outlaynm).(leabra.LeabraLayer).AsLeabra()

	// CosDiff calculates the cosine diff between ActM and ActP
	// MSE calculates the sum squared error and the mean squared error for the OutLay
	ss.TrlCosDiff = float64(outLay.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		if ss.HiddenType == "shared" && ss.TestEnv.Trial.Cur >= 0 && ss.TestEnv.Trial.Cur < 105 {
			ss.ShSumSSE += ss.TrlSSE
			ss.ShSumAvgSSE += ss.TrlAvgSSE
			ss.ShSumCosDiff += ss.TrlCosDiff
			if ss.TrlSSE != 0 {
				ss.ShCntErr++
			}
		}
		if ss.HiddenType == "unique" && ss.TestEnv.Trial.Cur >= 0 && ss.TestEnv.Trial.Cur < 105 {
			ss.UnSumSSE += ss.TrlSSE
			ss.UnSumAvgSSE += ss.TrlAvgSSE
			ss.UnSumCosDiff += ss.TrlCosDiff
			if ss.TrlSSE != 0 {
				ss.UnCntErr++
			}
		}
	}
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	curTrial := ss.TrainEnv.Trial.Cur
	//fmt.Println(curTrial)
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
func (ss *Sim) TestTrial(returnOnChg bool, slptest bool) {

	ss.TestEnv.Step()

	//fmt.Println(ss.TestEnv.Trial.Cur)

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView("test")
		}
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train

	// Setting up train trial layer input/target chnages in this block
	f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
	f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
	f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
	f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
	f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
	classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
	codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()

	outlay := ""

	switch ss.HiddenFeature {
	case "1":
		outlay = f1.Name()
	case "2":
		outlay = f2.Name()
	case "3":
		outlay = f3.Name()
	case "4":
		outlay = f4.Name()
	case "5":
		outlay = f5.Name()
	case "classname":
		outlay = classname.Name()
	case "codename":
		outlay = codename.Name()
	}
	ss.TrialStats(true, outlay) // !accumulate

	if slptest && ss.SlpTstWrtOut {
		dirtrlstats := "output/" + "slp_tst/" + fmt.Sprint(ss.DirSeed) + "/"
		if _, err := os.Stat(filepath.FromSlash(dirtrlstats)); os.IsNotExist(err) {
			os.MkdirAll(filepath.FromSlash(dirtrlstats), os.ModePerm)
		}

		filetrlstats, _ := os.OpenFile(filepath.FromSlash(dirtrlstats+"/"+"trlststats"+fmt.Sprint(ss.RndSeed)+".csv"),
			os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		defer filetrlstats.Close()
		writertrlstats := csv.NewWriter(filetrlstats)
		defer writertrlstats.Flush()

		if ss.TestEnv.Trial.Cur == 0 && ss.TrainEnv.Epoch.Cur == 1 {
			headers := []string{"Seed", "TrialName", "TrialSSE", "TrialAvgSSE", "TrialCor", "TrialHidType", "TrialHiddenFeature"}
			writertrlstats.Write(headers)
		}

		valueStr := []string{fmt.Sprint(ss.RndSeed), fmt.Sprint(ss.TestEnv.TrialName.Cur), fmt.Sprint(ss.TrlSSE),
			fmt.Sprint(ss.TrlAvgSSE), fmt.Sprint(ss.TrlSSE == 0), fmt.Sprint(ss.HiddenType),
			fmt.Sprint(ss.HiddenFeature)}
		writertrlstats.Write(valueStr)

	}
}

// TestItem tests given item which is at given index in test item list
// Currently Testitem will not do trialstats accum
func (ss *Sim) TestItem(idx int) {

	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll(slptest bool) {

	ss.TestNm = "Train Sat Permutations"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestSat)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)

	ss.HiddenType = ""
	ss.HiddenFeature = ""
	ss.UnTrlNum = 0
	ss.ShTrlNum = 0

	// Setting up train trial layer input/target chnages in this block
	f1 := ss.Net.LayerByName("F1").(leabra.LeabraLayer).AsLeabra()
	f2 := ss.Net.LayerByName("F2").(leabra.LeabraLayer).AsLeabra()
	f3 := ss.Net.LayerByName("F3").(leabra.LeabraLayer).AsLeabra()
	f4 := ss.Net.LayerByName("F4").(leabra.LeabraLayer).AsLeabra()
	f5 := ss.Net.LayerByName("F5").(leabra.LeabraLayer).AsLeabra()
	classname := ss.Net.LayerByName("ClassName").(leabra.LeabraLayer).AsLeabra()
	codename := ss.Net.LayerByName("CodeName").(leabra.LeabraLayer).AsLeabra()
	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	ctx := ss.Net.LayerByName("CTX").(leabra.LeabraLayer).AsLeabra()
	pca1 := ss.Net.LayerByName("pCA1").(leabra.LeabraLayer).AsLeabra()
	dca1 := ss.Net.LayerByName("dCA1").(leabra.LeabraLayer).AsLeabra()

	lesion := 1
	if slptest {
		lesion = 5
	} else {
		lesion = 1
	}

	for k := 0; k < lesion; k++ {
		if k == 1 {
			ctx.SetOff(true)
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}

		if k == 2 {
			dg.SetOff(true)
			ca3.SetOff(true)
			pca1.SetOff(true)
			dca1.SetOff(true)
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}

		if k == 3 {
			pca1.SetOff(true) //pca1 off  test
			ctx.SetOff(true)
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}

		if k == 4 {
			dca1.SetOff(true) //dca1 off  test
			ctx.SetOff(true)
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}

		if k == 5 { // pca1 -> ctx on, but pca1 -> per lys off
			dca1.SetOff(true) //dca1 off  test
			ctx.SetOff(false)

			perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
			for _, ly := range perlys {
				lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
				lyc.RcvPrjns.SendName("pCA1").(*hip.CHLPrjn).WtScale.Abs = 0
			}

			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}

		if k == 6 { // dca1 -> ctx on, but dca1 -> per lys off
			pca1.SetOff(true) // dca1 off  test
			ctx.SetOff(false) // i.e. pca1 + ctx

			perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
			for _, ly := range perlys {
				lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
				lyc.RcvPrjns.SendName("dCA1").(*hip.CHLPrjn).WtScale.Abs = 0
				lyc.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).WtScale.Abs = 0
			}

			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		}

		for i := 0; i < 7; i++ { // i < 7
			for j := 0; j < 15; j++ {

				switch i {
				case 0:
					f1.SetType(emer.Target)
					f1.UpdateExtFlags()
					ss.HiddenFeature = "1"
				case 1:
					f2.SetType(emer.Target)
					f2.UpdateExtFlags()
					ss.HiddenFeature = "2"
				case 2:
					f3.SetType(emer.Target)
					f3.UpdateExtFlags()
					ss.HiddenFeature = "3"
				case 3:
					f4.SetType(emer.Target)
					f4.UpdateExtFlags()
					ss.HiddenFeature = "4"
				case 4:
					f5.SetType(emer.Target)
					f5.UpdateExtFlags()
					ss.HiddenFeature = "5"
				case 5:
					classname.SetType(emer.Target)
					classname.UpdateExtFlags()
					ss.HiddenFeature = "classname"
				case 6:
					codename.SetType(emer.Target)
					codename.UpdateExtFlags()
					ss.HiddenFeature = "codename"
				}

				name := ss.TestEnv.TrialName.Cur

				ss.HiddenType = "shared"
				for n, feature := range name {
					if (i < 5) && (string(feature) == "4" || string(feature) == "5" || string(feature) == "6") && (n == i) { // checking here if there is a unique feature and if it is the currently hidden one
						ss.HiddenType = "unique"
						break
					} else if i == 6 {
						ss.HiddenType = "unique"
						break
					}
				}

				ss.TestTrial(true, slptest) // return on chg

				ss.LogTstTrl(ss.TstTrlLog)

				f1.SetType(emer.Input)
				f1.UpdateExtFlags()
				f2.SetType(emer.Input)
				f2.UpdateExtFlags()
				f3.SetType(emer.Input)
				f3.UpdateExtFlags()
				f4.SetType(emer.Input)
				f4.UpdateExtFlags()
				f5.SetType(emer.Input)
				f5.UpdateExtFlags()
				classname.SetType(emer.Input)
				classname.UpdateExtFlags()
				codename.SetType(emer.Input)
				codename.UpdateExtFlags()

				_, _, chg := ss.TestEnv.Counter(env.Epoch)
				if chg || ss.StopNow {
					break
				}
				//fmt.Println(outlay)

				if ss.HiddenType == "unique" {
					ss.UnTrlNum++
				} else {
					ss.ShTrlNum++
				}

				ss.Net.GScaleFmAvgAct() // update computed scaling factors
				ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

			}
		}

		dg.SetOff(false)
		ca3.SetOff(false)
		ctx.SetOff(false)
		pca1.SetOff(false)
		dca1.SetOff(false)

		perlys := []string{"F1", "F2", "F3", "F4", "F5", "CodeName", "ClassName"}
		for _, ly := range perlys {
			lyc := ss.Net.LayerByName(ly).(*leabra.Layer).AsLeabra()
			lyc.RcvPrjns.SendName("dCA1").(*hip.CHLPrjn).WtScale.Abs = 1
			lyc.SndPrjns.RecvName("dCA1").(*hip.CHLPrjn).WtScale.Abs = 1
			lyc.RcvPrjns.SendName("pCA1").(*hip.CHLPrjn).WtScale.Abs = 1
		}

		ss.Net.GScaleFmAvgAct() // update computed scaling factors
		ss.Net.InitGInc()       // scaling params change, so need to recompute all netins

	}

	ss.LogTstEpc(ss.TstEpcLog)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll(false)
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
	ss.OpenPat(ss.TrainSat, "train_sats.txt", "TrainSat", "Training Patterns")
	ss.OpenPat(ss.TestSat, "test_sats.txt", "TestSat", "Testing Patterns")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

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
	dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
	dt.SetCellString("HiddenType", row, ss.HiddenType)
	dt.SetCellString("HiddenFeature", row, ss.HiddenFeature)
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
		{"HiddenType", etensor.STRING, nil, nil},
		{"HiddenFeature", etensor.STRING, nil, nil},
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
	plt.SetColParams("HiddenType", true, true, 0, false, 0)
	plt.SetColParams("HiddenFeature", false, true, 0, false, 0)
	plt.SetColParams("SSE", true, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	return plt
}

func (ss *Sim) LogSlpCyc(dt *etable.Table, cyc int) {

	row := dt.Rows
	if cyc == 0 {
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

	epc := ss.TrainEnv.Epoch.Prv         // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Trial.Max) // number of trials in view
	shnt := float64(ss.ShTrlNum)
	unnt := float64(ss.UnTrlNum)

	// Computing Epc Shared/Unique feature learning metrics
	ss.EpcShSSE = ss.ShSumSSE / shnt
	ss.ShSumSSE = 0
	ss.EpcShAvgSSE = ss.ShSumAvgSSE / shnt
	ss.ShSumAvgSSE = 0
	ss.EpcShPctErr = float64(ss.ShCntErr) / shnt
	ss.ShCntErr = 0
	ss.EpcShPctCor = 1 - ss.EpcShPctErr
	ss.EpcShCosDiff = ss.ShSumCosDiff / shnt
	ss.ShSumCosDiff = 0
	ss.ShTrlNum = 0

	ss.EpcUnSSE = ss.UnSumSSE / unnt
	ss.UnSumSSE = 0
	ss.EpcUnAvgSSE = ss.UnSumAvgSSE / unnt
	ss.UnSumAvgSSE = 0
	ss.EpcUnPctErr = float64(ss.UnCntErr) / unnt
	ss.UnCntErr = 0
	ss.EpcUnPctCor = 1 - ss.EpcUnPctErr
	ss.EpcUnCosDiff = ss.UnSumCosDiff / unnt
	ss.UnSumCosDiff = 0
	ss.UnTrlNum = 0

	// Adding shared/unique metrics to log
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Total Trials", row, float64(nt))
	dt.SetCellFloat("Shared Trials", row, float64(shnt))
	dt.SetCellFloat("ShSSE", row, ss.EpcShSSE)
	dt.SetCellFloat("ShAvgSSE", row, ss.EpcShAvgSSE)
	dt.SetCellFloat("ShPctErr", row, ss.EpcShPctErr)
	dt.SetCellFloat("ShPctCor", row, ss.EpcShPctCor)
	dt.SetCellFloat("ShCosDiff", row, ss.EpcShCosDiff)
	dt.SetCellFloat("Unique Trials", row, float64(unnt))
	dt.SetCellFloat("UnSSE", row, ss.EpcUnSSE)
	dt.SetCellFloat("UnAvgSSE", row, ss.EpcUnAvgSSE)
	dt.SetCellFloat("UnPctErr", row, ss.EpcUnPctErr)
	dt.SetCellFloat("UnPctCor", row, ss.EpcUnPctCor)
	dt.SetCellFloat("UnCosDiff", row, ss.EpcUnCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()

	if ss.EpcUnSSE == 0 && ss.EpcShSSE == 0 {
		ss.ZError++
	} else {
		ss.ZError = 0
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
		{"Total Trials", etensor.INT64, nil, nil},
		{"Shared Trials", etensor.INT64, nil, nil},
		{"ShSSE", etensor.FLOAT64, nil, nil},
		{"ShAvgSSE", etensor.FLOAT64, nil, nil},
		{"ShPctErr", etensor.FLOAT64, nil, nil},
		{"ShPctCor", etensor.FLOAT64, nil, nil},
		{"ShCosDiff", etensor.FLOAT64, nil, nil},
		{"Unique Trials", etensor.INT64, nil, nil},
		{"UnSSE", etensor.FLOAT64, nil, nil},
		{"UnAvgSSE", etensor.FLOAT64, nil, nil},
		{"UnPctErr", etensor.FLOAT64, nil, nil},
		{"UnPctCor", etensor.FLOAT64, nil, nil},
		{"UnCosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("ShSSE", true, true, 0, false, 0)
	plt.SetColParams("ShAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("ShPctErr", false, true, 0, true, 1)
	plt.SetColParams("ShPctCor", true, true, 0, true, 1)
	plt.SetColParams("ShCosDiff", false, true, 0, true, 1)
	plt.SetColParams("UnSSE", true, true, 0, false, 0)
	plt.SetColParams("UnAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("UnPctErr", false, true, 0, true, 1)
	plt.SetColParams("UnPctCor", true, true, 0, true, 1)
	plt.SetColParams("UnCosDiff", false, true, 0, true, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", false, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur

	row := dt.Rows
	if ss.TestNm == "Train Sat Permutations" && trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestNm", row, ss.TestNm)
	dt.SetCellFloat("Trial", row, float64(row))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellString("HiddenType", row, ss.HiddenType)
	dt.SetCellString("HiddenFeature", row, ss.HiddenFeature)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {

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
		{"HiddenType", etensor.STRING, nil, nil},
		{"HiddenFeature", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("TestNm", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("HiddenType", true, true, 0, false, 0)
	plt.SetColParams("HiddenFeature", false, true, 0, false, 0)
	plt.SetColParams("SSE", true, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", false, true, 0, true, .5)
	}

	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {

	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv
	nt := float64(ss.TestEnv.Trial.Max)
	shnt := float64(78)
	unnt := float64(27)

	// Computing Epc Shared/Unique feature learning metrics
	ss.EpcShSSE = ss.ShSumSSE / shnt
	ss.ShSumSSE = 0
	ss.EpcShAvgSSE = ss.ShSumAvgSSE / shnt
	ss.ShSumAvgSSE = 0
	ss.EpcShPctErr = float64(ss.ShCntErr) / shnt
	ss.ShCntErr = 0
	ss.EpcShPctCor = 1 - ss.EpcShPctErr
	ss.EpcShCosDiff = ss.ShSumCosDiff / shnt
	ss.ShSumCosDiff = 0
	ss.ShTrlNum = 0

	ss.EpcUnSSE = ss.UnSumSSE / unnt
	ss.UnSumSSE = 0
	ss.EpcUnAvgSSE = ss.UnSumAvgSSE / unnt
	ss.UnSumAvgSSE = 0
	ss.EpcUnPctErr = float64(ss.UnCntErr) / unnt
	ss.UnCntErr = 0
	ss.EpcUnPctCor = 1 - ss.EpcUnPctErr
	ss.EpcUnCosDiff = ss.UnSumCosDiff / unnt
	ss.UnSumCosDiff = 0
	ss.UnTrlNum = 0

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Total Trials", row, float64(nt))
	dt.SetCellFloat("Shared Trials", row, float64(shnt))
	dt.SetCellFloat("ShSSE", row, ss.EpcShSSE)
	dt.SetCellFloat("ShAvgSSE", row, ss.EpcShAvgSSE)
	dt.SetCellFloat("ShPctErr", row, ss.EpcShPctErr)
	dt.SetCellFloat("ShPctCor", row, ss.EpcShPctCor)
	dt.SetCellFloat("ShCosDiff", row, ss.EpcShCosDiff)
	dt.SetCellFloat("Unique Trials", row, float64(unnt))
	dt.SetCellFloat("UnSSE", row, ss.EpcUnSSE)
	dt.SetCellFloat("UnAvgSSE", row, ss.EpcUnAvgSSE)
	dt.SetCellFloat("UnPctErr", row, ss.EpcUnPctErr)
	dt.SetCellFloat("UnPctCor", row, ss.EpcUnPctCor)
	dt.SetCellFloat("UnCosDiff", row, ss.EpcUnCosDiff)

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
		{"Total Trials", etensor.INT64, nil, nil},
		{"Shared Trials", etensor.INT64, nil, nil},
		{"ShSSE", etensor.FLOAT64, nil, nil},
		{"ShAvgSSE", etensor.FLOAT64, nil, nil},
		{"ShPctErr", etensor.FLOAT64, nil, nil},
		{"ShPctCor", etensor.FLOAT64, nil, nil},
		{"ShCosDiff", etensor.FLOAT64, nil, nil},
		{"Unique Trials", etensor.INT64, nil, nil},
		{"UnSSE", etensor.FLOAT64, nil, nil},
		{"UnAvgSSE", etensor.FLOAT64, nil, nil},
		{"UnPctErr", etensor.FLOAT64, nil, nil},
		{"UnPctCor", etensor.FLOAT64, nil, nil},
		{"UnCosDiff", etensor.FLOAT64, nil, nil},
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Sleep-replay Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("ShSSE", true, true, 0, false, 0)
	plt.SetColParams("ShAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("ShPctErr", false, true, 0, true, 1)
	plt.SetColParams("ShPctCor", true, true, 0, true, 1)
	plt.SetColParams("ShCosDiff", false, true, 0, true, 1)
	plt.SetColParams("UnSSE", true, true, 0, false, 0)
	plt.SetColParams("UnAvgSSE", false, true, 0, false, 0)
	plt.SetColParams("UnPctErr", false, true, 0, true, 1)
	plt.SetColParams("UnPctCor", true, true, 0, true, 1)
	plt.SetColParams("UnCosDiff", false, true, 0, true, 1)

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

	gi.SetAppName("SNS22_Sim1")
	gi.SetAppAbout(`This is a Sleep-replay model developed in Leabra. See <a href="https://github.com/schapirolab/SinghNormanSchapiro_PNAS22"> on GitHub</a>.</p>`)

	win := gi.NewMainWindow("SNS22_Sim1", "SNS22_Sim1", width, height)
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
	nv.Params.MaxRecs = 200
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
			ss.TestTrial(false, false) // don't return on trial -- wrap
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
	flag.IntVar(&ss.MaxRuns, "runs", 100, "number of runs to do (note that MaxEpcs is in paramset)")
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
